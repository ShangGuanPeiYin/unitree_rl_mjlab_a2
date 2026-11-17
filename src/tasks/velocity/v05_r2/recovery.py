"""v0.5-r2 schema-7 checkpoints, durable ledger, evaluation transactions and recovery.

Owned responsibilities (SDS v0.5-r2 7.4/7.5/7.7):

* schema-7 checkpoint builder/validator (:func:`build_checkpoint`,
  :func:`validate_checkpoint`, :func:`preload_checkpoint`); schema <= 6, r1
  full checkpoints, r1 initialization containers and the old/ r2 smoke
  containers are refused, never migrated and never opened with ``strict=False``.
* the schema-7 model body must carry the r2 route bypass: exactly
  ``actor.route_fusion.weight [512,3]``, ``actor.route_fusion.bias [512]`` and
  ``critic.0.weight [512,211]``; a missing/short-keyed model is rejected.
* one durable, fsync-per-control-step hash-chained JSONL ledger
  (:class:`AttemptLedger`) whose events are the only charge evidence.  The
  ledger *derives* the v0.5 curriculum/streak/pending-transition state, so an
  evaluation consumption and its streak change are literally one event.
* the evaluation transaction state machine ``planned -> result_committed ->
  consumed`` with the unique key
  ``(run_id, mode, stage, global_update, kind, checkpoint_sha, case_sha)``.
* newest-complete-boundary checkpoint selection by ledger order (never by file
  mtime) and the atomic state/checkpoint-reference reconciliation.
* the unique SDS 7.3 restore order (:class:`RecoverySequence`) and monotonic
  worker attempt directories (:func:`claim_worker_directory`).

Nothing here imports an older version package: the schema-5 v0.4 validator and
the running r1 ``v05/recovery.py`` are not reused (they reject every r2
envelope by design) and the reward/window state is never restored.

SDS 7.3 budget: ``total_budget``/``update_cap`` are read from the single
``spec.mode_budget(mode)`` source (pilot 2500, formal 30000); :func:`audit_budget`
is the resume-side check that a checkpoint's learning count, its curriculum
spent table and the mode cap agree.  The checkpoint never stores a rollout, so
``terminal_observation['route_error']`` is traceable through the evaluation
evidence that consumes that checkpoint (SDS 4.5/4.8); :func:`route_evidence_contract`
records that split and :func:`validate_checkpoint` rejects a foreign
observation contract.
"""
from __future__ import annotations

from copy import deepcopy
import fcntl
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile

import numpy as np
import torch

from .spec import (ARCHITECTURE, CODE_VERSION, CONTRACT_MODES, DECIMATION, INIT_PATH,
                   INIT_SHA256, MODE_BUDGETS, OBSERVATION, PPO_EPOCHS, PPO_MINIBATCHES,
                   PPO_ROLLOUT, PROTOCOL_ID, RECIPE_VERSION, SCHEMA_VERSION, SDS_REVISION,
                   SMOKE_CONTAINER, STAGE_BY_ID, STAGE_ORDER, TASK, contract,
                   contract_sha256, mode_budget, recipe_config, sha256)

HEX = re.compile(r'^[0-9a-f]{64}$')
ADAM_STEPS_PER_UPDATE = PPO_EPOCHS * PPO_MINIBATCHES
RESUME_MODE = 'new_episode'
REWARD_STATE = 'new_episode_no_half_trajectory'
EVALUATION_KINDS = ('diagnostic', 'validation', 'flat_retention', 'perturbed', 'test')
EVALUATION_STATES = ('planned', 'result_committed', 'consumed')
TASK_KEY_FIELDS = ('run_id', 'mode', 'stage', 'global_update', 'kind',
                   'checkpoint_sha256', 'case_sha256')

CHECKPOINT_KEYS = frozenset({'schema', 'identity', 'contract_sha256', 'model', 'optimizer',
                             'normalizer', 'learning', 'rng', 'curriculum', 'ledger',
                             'evaluation', 'source', 'recipe', 'resume_mode'})
IDENTITY_KEYS = frozenset({'task', 'recipe', 'code_version', 'protocol', 'sds_revision'})
LEARNING_KEYS = frozenset({'update', 'optimizer_step', 'control_samples', 'physics_substeps'})
RNG_KEYS = frozenset({'python', 'numpy', 'torch', 'cuda', 'env'})
CURRICULUM_KEYS = frozenset({'stage', 'stage_iteration', 'spent', 'streak', 'pending_transition'})
CURRICULUM_STREAK_KEYS = frozenset({'gate', 'collapse'})
PENDING_TRANSITION_KEYS = frozenset({'from_stage', 'to_stage', 'at_update', 'requires',
                                     'at_checkpoint'})
LEDGER_STATE_KEYS = frozenset({'tasks', 'results', 'collapse_streak', 'gate_streak',
                               'anchor', 'qualified_update', 'pending_transition'})
EVALUATION_KEYS = frozenset({'consumed', 'latest_diagnostic_update'})
SOURCE_KEYS = frozenset({'init_path', 'init_sha256', 'resume', 'mode', 'seed', 'device'})
TASK_RECORD_KEYS = frozenset({'state', 'run_id', 'mode', 'stage', 'global_update', 'kind',
                              'checkpoint_sha256', 'case_sha256', 'planned_wall'})
RESULT_KEYS = frozenset({'result_path', 'result_sha256', 'report_path', 'report_sha256',
                         'evidence_sha256', 'summary_sha256', 'committed_wall'})
CONSUMPTION_KEYS = frozenset({'outcome', 'passed', 'collapse_hit', 'result_sha256',
                              'consumed_wall', 'streak'})
RESULT_RECORD_KEYS = frozenset({'task_key', 'kind', 'stage', 'global_update', 'outcome',
                                'passed', 'gate_streak', 'collapse_streak'})
LEDGER_HEADER_KEYS = frozenset({'schema', 'run_id', 'mode'})
LEDGER_SNAPSHOT_KEYS = frozenset({'header', 'events', 'seq', 'digest'})
#: SDS 7.5: the schema-7 model body must carry these exact r2 route tensors.
ROUTE_MODEL_KEYS = {
    'actor.route_fusion.weight': (512, 3),
    'actor.route_fusion.bias': (512,),
    'critic.0.weight': (512, 211),
}
#: SDS 1.1/7.5: the r2 observation contract is five keys and route_error is [3].
ROUTE_OBSERVATION_KEY = 'route_error'
ROUTE_OBSERVATION_SHAPE = [3]
#: Known smoke containers.  Neither the r1 ``v05-smoke-state-v1`` nor r2's own
#: ``v05-r2-smoke-state-v1`` is a pilot/formal resume source (SDS 7.5-3).
SMOKE_CONTAINER_IDS = ('v05-smoke-state-v1', SMOKE_CONTAINER)
#: r1/ETH migration evidence is not a resume checkpoint (SDS 7.5-4).
INITIALIZATION_SCOPE = 'initialization'
#: Full events kept in memory.  A v0.5 formal run writes 72 durable events per
#: update (48 control samples + 20 Adam steps + 4 boundary events), so an
#: unbounded in-memory history would reach several GB over 30000 updates.
#: Older prefixes stay on disk and are re-read only when a checkpoint anchor
#: older than this window has to be audited.
RETAINED_EVENTS = 4096


# --------------------------------------------------------------------- basics
def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'Invalid {name}')
    return value


def require_hex(value, name):
    if not isinstance(value, str) or not HEX.fullmatch(value):
        raise ValueError(f'{name} must be a 64-hex SHA256 string')
    return value


def finite_state(value, name='state'):
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError(f'Nonfinite {name}')
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_state(item, f'{name}.{key}')
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            finite_state(item, f'{name}.{index}')


def controls_per_update(num_envs):
    return PPO_ROLLOUT * integer(num_envs, 'num_envs', 1)


def learning_state(update, *, num_envs=512):
    """The exact complete learning boundary for ``update`` (release 1.3: no refund)."""
    integer(update, 'completed PPO updates')
    controls = controls_per_update(num_envs) * update
    return dict(update=update, optimizer_step=ADAM_STEPS_PER_UPDATE * update,
                control_samples=controls, physics_substeps=DECIMATION * controls)


def validate_learning(state, *, num_envs=None):
    """Validate the schema-7 ``learning`` block; returns the update count."""
    if not isinstance(state, dict) or set(state) != LEARNING_KEYS:
        raise ValueError('Complete learning identity required')
    update = integer(state['update'], 'completed PPO updates')
    if not isinstance(state['optimizer_step'], int) or isinstance(state['optimizer_step'], bool):
        raise ValueError('Invalid optimizer_step')
    if state['optimizer_step'] != ADAM_STEPS_PER_UPDATE * update:
        raise ValueError('optimizer_step must be exactly 20 Adam steps per complete update')
    controls = integer(state['control_samples'], 'control_samples')
    physics = integer(state['physics_substeps'], 'physics_substeps')
    if physics != DECIMATION * controls:
        raise ValueError('physics_substeps must be exactly 4*control_samples')
    if update:
        if controls % (update * PPO_ROLLOUT):
            raise ValueError('control_samples is not a whole number of updates')
        actual = controls // (update * PPO_ROLLOUT)
        if actual < 1 or (num_envs is not None and actual != num_envs):
            raise ValueError('control_samples implies an unapproved environment count')
    elif controls:
        raise ValueError('zero completed updates must have zero control samples')
    return update


# ----------------------------------------------------------------- curriculum
def genesis_curriculum():
    return dict(stage=STAGE_ORDER[0], stage_iteration=0,
                spent={stage: 0 for stage in STAGE_ORDER},
                streak=dict(gate=0, collapse=0), pending_transition=None)


def stage_budgets(mode='formal'):
    """Effective per-stage caps from the single ``spec.mode_budget`` source.

    ``smoke`` deliberately has an empty stage table (a fixed two-update
    engineering cap, SDS 7.3); its per-stage structural check uses the frozen
    stage table while ``total_budget`` stays the smoke cap.  ``probe`` borrows
    the formal stage table from the same source.
    """
    record = mode_budget(mode)
    stages = record['stages']
    # A mode may cap only a prefix of the stage table (pilot: F0/P1; smoke: none).
    # Stages outside that prefix keep the frozen formal structural cap while the
    # *total* cap still comes from ``mode_budget(mode)`` (SDS 7.3).
    formal = mode_budget('formal')['stages']
    return {name: int(stages.get(name, formal[name])) for name in STAGE_ORDER}


def total_budget(mode='formal'):
    """Effective update cap (pilot 2500 / formal 30000) from ``mode_budget``."""
    return int(mode_budget(mode)['total'])


def validate_curriculum(state, *, learning=None, mode='formal'):
    """Structural and budget validation of the schema-7 ``curriculum`` block.

    The per-stage and total caps come from ``spec.mode_budget(mode)`` (SDS 7.3),
    so a pilot P1 spent of 500 validates as complete instead of being compared
    with the formal 1000.
    """
    if not isinstance(state, dict) or set(state) != CURRICULUM_KEYS:
        raise ValueError('Complete curriculum identity required')
    stage = state['stage']
    if stage not in STAGE_BY_ID:
        raise ValueError('Unknown v0.5 stage in curriculum')
    iteration = integer(state['stage_iteration'], 'stage_iteration')
    spent = state['spent']
    if not isinstance(spent, dict) or set(spent) != set(STAGE_ORDER):
        raise ValueError('per-stage spent must cover exactly the ten stages')
    budgets = stage_budgets(mode)
    for name, value in spent.items():
        integer(value, f'spent[{name}]')
        if value > budgets[name]:
            raise ValueError('stage budget exceeded in curriculum')
    if sum(spent.values()) > total_budget(mode):
        raise ValueError('total budget exceeded in curriculum')
    if spent[stage] != iteration:
        raise ValueError('stage_iteration differs from per-stage spent')
    streak = state['streak']
    if not isinstance(streak, dict) or set(streak) != CURRICULUM_STREAK_KEYS:
        raise ValueError('curriculum streak must hold gate and collapse counts')
    integer(streak['gate'], 'gate streak')
    integer(streak['collapse'], 'collapse streak')
    pending = state['pending_transition']
    if pending is not None:
        if not isinstance(pending, dict) or set(pending) != PENDING_TRANSITION_KEYS:
            raise ValueError('pending_transition schema mismatch')
        if pending['from_stage'] != stage or pending['from_stage'] not in STAGE_BY_ID:
            raise ValueError('pending transition must originate at the current stage')
        if pending['to_stage'] is not None:
            index = STAGE_ORDER.index(stage)
            if index + 1 >= len(STAGE_ORDER) or pending['to_stage'] != STAGE_ORDER[index + 1]:
                raise ValueError('pending transition must target the next stage only')
        if pending['requires'] not in (None, 'perturbed'):
            raise ValueError('unknown pending transition requirement')
        integer(pending['at_update'], 'pending transition update')
        if not isinstance(pending['at_checkpoint'], str) or not HEX.fullmatch(pending['at_checkpoint']):
            raise ValueError('pending transition must bind a checkpoint SHA256')
    if learning is not None:
        update = validate_learning(learning)
        if sum(spent.values()) != update:
            raise ValueError('per-stage spent does not sum to completed PPO updates')
    return state


def advance_curriculum(state, mode='formal'):
    """Apply the one legal stage transition; returns a new copy."""
    validate_curriculum(state, mode=mode)
    pending = state['pending_transition']
    if pending is None or pending['to_stage'] is None:
        raise ValueError('No stage transition is pending')
    if pending['at_update'] != sum(state['spent'].values()):
        raise ValueError('Pending transition is not at the current complete boundary')
    new = deepcopy(state)
    new['stage'] = pending['to_stage']
    new['stage_iteration'] = 0
    new['streak'] = dict(gate=0, collapse=0)
    new['pending_transition'] = None
    return new


def updated_curriculum(state, *, num_envs, mode='formal'):
    """Curriculum after one complete update in the current stage."""
    validate_curriculum(state, mode=mode)
    new = deepcopy(state)
    new['stage_iteration'] += 1
    new['spent'][new['stage']] += 1
    if new['spent'][new['stage']] > stage_budgets(mode)[new['stage']]:
        raise ValueError('Stage budget exhausted; curriculum cannot charge another update')
    if sum(new['spent'].values()) > total_budget(mode):
        raise ValueError('Total budget exhausted; curriculum cannot charge another update')
    integer(num_envs, 'num_envs', 1)
    return new


def audit_budget(learning, curriculum, *, mode='formal'):
    """SDS 7.3 resume-side budget audit against the single mode source.

    Checks ``learning.update <= mode_budget(mode).total``, the current stage's
    spent ``<=`` that stage's cap, and ``sum(spent) <= total``.  Returns the
    audit block written into result/report metadata; a mismatch is a hard
    ``ValueError`` (the caller maps it to ``recovery_blocked``).
    """
    update = validate_learning(learning)
    validate_curriculum(curriculum, learning=learning, mode=mode)
    budgets = stage_budgets(mode)
    total = total_budget(mode)
    spent = curriculum['spent']
    if update > total:
        raise ValueError(f'completed updates {update} exceed the {mode} cap {total}')
    if spent[curriculum['stage']] > budgets[curriculum['stage']]:
        raise ValueError('current stage spent exceeds the mode stage budget')
    if sum(spent.values()) > total:
        raise ValueError('curriculum spent exceeds the mode total budget')
    return dict(mode=mode, total_budget=total, update_cap=total,
                effective_cap_source='mode_budget', update=update,
                stage=curriculum['stage'], stage_budget=budgets[curriculum['stage']],
                stage_spent=spent[curriculum['stage']],
                total_spent=sum(spent.values()), stage_budgets=budgets)


# ------------------------------------------------------------------------ rng
def capture_rng(env):
    if not isinstance(getattr(env, 'rng', None), torch.Generator):
        raise ValueError('Dedicated environment Generator required')
    return dict(env=env.rng.get_state().clone(), python=random.getstate(),
                numpy=np.random.get_state(), torch=torch.get_rng_state().clone(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def validate_rng(state):
    if not isinstance(state, dict) or set(state) != RNG_KEYS or not isinstance(state['cuda'], list):
        raise ValueError('Complete training RNG required')
    for item in [state['env'], state['torch'], *state['cuda']]:
        if (not isinstance(item, torch.Tensor) or item.dtype != torch.uint8 or item.ndim != 1
                or item.numel() == 0 or item.device.type != 'cpu'):
            raise ValueError('Invalid raw RNG tensor')
    try:
        random.Random().setstate(state['python'])
        np.random.RandomState().set_state(state['numpy'])
        torch.Generator().set_state(state['torch'])
        # CUDA states use a device-specific format; CPU validation never allocates.
        if not state['cuda']:
            torch.Generator().set_state(state['env'])
    except Exception as error:  # noqa: BLE001 - any decoder failure means invalid state
        raise ValueError('Invalid training RNG state') from error


def restore_rng(state, env):
    validate_rng(state)
    if not isinstance(getattr(env, 'rng', None), torch.Generator):
        raise ValueError('Dedicated environment Generator required')
    if state['cuda'] and (not torch.cuda.is_available()
                          or len(state['cuda']) != torch.cuda.device_count()):
        raise ValueError('CUDA RNG device set mismatch')
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
    env.rng.set_state(state['env'])


# ------------------------------------------------------------------- optimizer
def validate_optimizer_state(state, parameters, learning, *, cfg=None):
    """Adam state must be a complete 20-step boundary with unchanged settings.

    ``parameters`` is a sequence of tensors (``list(model.parameters())``), so a
    caller without the full model can still validate the optimizer body.
    """
    from .ppo import PPOConfig
    cfg = PPOConfig() if cfg is None else cfg
    update = validate_learning(learning)
    parameters = list(parameters)
    if not parameters:
        raise ValueError('No model parameters to validate against Adam state')
    finite_state(state, 'optimizer')
    template = torch.optim.Adam(parameters, lr=cfg.learning_rate).state_dict()
    if (not isinstance(state, dict) or set(state) != set(template)
            or not isinstance(state['state'], dict) or not isinstance(state['param_groups'], list)
            or len(state['param_groups']) != 1):
        raise ValueError('Adam schema mismatch')
    group, expected = state['param_groups'][0], template['param_groups'][0]
    if not isinstance(group, dict) or set(group) != set(expected):
        raise ValueError('Adam parameter group schema mismatch')
    if group['params'] != expected['params']:
        raise ValueError('Adam parameter order mismatch')
    if (isinstance(group['lr'], bool) or not isinstance(group['lr'], (int, float))
            or not math.isfinite(group['lr'])
            or not cfg.min_lr <= group['lr'] <= cfg.max_lr):
        raise ValueError('Adam learning rate outside the approved [1e-5, 1e-4] range')
    for key, value in expected.items():
        if key not in ('lr', 'params') and group[key] != value:
            raise ValueError(f'Adam algorithm setting changed: {key}')
    required = set(expected['params']) if update else set()
    if set(state['state']) != required:
        raise ValueError('Missing or extra complete Adam parameter state')
    for index, moments in state['state'].items():
        if not isinstance(moments, dict) or set(moments) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise ValueError('Adam moments schema mismatch')
        step = moments['step']
        if not isinstance(step, torch.Tensor) or step.ndim != 0:
            raise ValueError('Adam step must be a scalar tensor')
        if step.item() != ADAM_STEPS_PER_UPDATE * update:
            raise ValueError('Adam step differs from the complete learning boundary')
        for key in ('exp_avg', 'exp_avg_sq'):
            value = moments[key]
            if (not isinstance(value, torch.Tensor) or value.shape != parameters[index].shape
                    or value.dtype != parameters[index].dtype):
                raise ValueError('Adam moment shape/dtype mismatch')
        if (moments['exp_avg_sq'] < 0).any():
            raise ValueError('Negative Adam second moment')


# ------------------------------------------------------------------ checkpoint
def identity_block():
    return dict(task=TASK, recipe=RECIPE_VERSION, code_version=CODE_VERSION,
                protocol=PROTOCOL_ID, sds_revision=SDS_REVISION)


def architecture():
    """The frozen r2 architecture identity (SDS 1.1)."""
    return ARCHITECTURE


def route_evidence_contract():
    """Where the r2 route observation is contract-bound and where it is not.

    A schema-7 checkpoint persists model/Adam/normalizer/learning/curriculum/
    ledger/evaluation/source/recipe only; it never persists a rollout (SDS 4.4-8).
    ``terminal_observation['route_error']`` is therefore traceable through the
    evaluation evidence indexed by the ledger task key
    ``(..., checkpoint_sha256, case_sha256)`` and the episode ``route_final`` /
    ``route_initial`` fields, not through the checkpoint body.  This function is
    the explicit record of that split for the audit report.
    """
    return dict(observation_keys=list(OBSERVATION), route_key=ROUTE_OBSERVATION_KEY,
                route_shape=ROUTE_OBSERVATION_SHAPE, architecture=architecture(),
                model_keys=sorted(ROUTE_MODEL_KEYS),
                checkpoint_persists_rollout=False,
                terminal_route_evidence='evaluation_npz_and_episode_route_final',
                normalizer_dimensions={'actor_eth': 45, 'critic': 80})


def validate_observation_contract():
    """Fail closed when the spec observation is not the frozen five-key r2 one."""
    if not isinstance(OBSERVATION, dict):
        raise ValueError('spec OBSERVATION must be a mapping')
    if set(OBSERVATION) != {'proprio', 'history', 'heightmap', 'privileged',
                            ROUTE_OBSERVATION_KEY}:
        raise ValueError(f'spec OBSERVATION must be exactly the five r2 keys, got '
                         f'{sorted(OBSERVATION)}')
    if list(OBSERVATION[ROUTE_OBSERVATION_KEY]) != ROUTE_OBSERVATION_SHAPE:
        raise ValueError('spec route_error observation must be [3]')
    if architecture() != 'v05-route-state-v1':
        raise ValueError('schema-7 architecture must be v05-route-state-v1')
    return True


def source_block(*, mode='formal', seed=42, device='cuda:0', resume=None):
    if mode not in CONTRACT_MODES:
        raise ValueError(f'mode must be one of {CONTRACT_MODES}')
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 1:
        raise ValueError('seed must be a positive integer')
    if not isinstance(device, str) or not device:
        raise ValueError('device must be a non-empty string')
    resume = [] if resume is None else deepcopy(resume)
    if not isinstance(resume, list):
        raise ValueError('resume lineage must be a list')
    return dict(init_path=INIT_PATH, init_sha256=INIT_SHA256, resume=resume,
                mode=mode, seed=seed, device=device)


_CONTRACT_SHA_CACHE = {}


def contract_sha_for(source):
    """SHA256 of the v0.5 source/recipe contract for one execution identity.

    The source closure is parsed once per build parameters per process: a running
    training process cannot change its own source, and re-walking the AST for
    every checkpoint validation would dominate save time.
    """
    source = source_block(**{key: source[key] for key in ('mode', 'seed', 'device')})
    key = (source['mode'], source['seed'], source['device'])
    if key not in _CONTRACT_SHA_CACHE:
        _CONTRACT_SHA_CACHE[key] = contract_sha256(
            contract(mode=source['mode'], seed=source['seed'], device=source['device']))
    return _CONTRACT_SHA_CACHE[key]


def build_checkpoint(*, model_state, optimizer_state, normalizer, learning, rng,
                     curriculum, ledger, evaluation=None, source=None, mode='formal',
                     seed=42, device='cuda:0', resume=None, parameters=None,
                     model_factory=None):
    """Assemble an exact schema-7 checkpoint envelope (runner calls this)."""
    source = source_block(mode=mode, seed=seed, device=device,
                          resume=source['resume'] if source else resume)
    validate_observation_contract()
    validate_route_model_state(model_state)
    if isinstance(ledger, AttemptLedger):
        task_state = ledger.checkpoint_task_state()
        if evaluation is None:
            evaluation = ledger.checkpoint_evaluation()
    else:
        task_state = deepcopy(ledger)
    state = dict(
        schema=SCHEMA_VERSION,
        identity=identity_block(),
        contract_sha256=contract_sha_for(source),
        model=model_state,
        optimizer=optimizer_state,
        normalizer=normalizer,
        learning=deepcopy(learning),
        rng=deepcopy(rng),
        curriculum=deepcopy(curriculum),
        ledger=task_state,
        evaluation=deepcopy(evaluation),
        source=source,
        recipe=recipe_config(),
        resume_mode=RESUME_MODE,
    )
    validate_checkpoint(state, parameters=parameters, model_factory=model_factory)
    return state


def _identity_mismatch(value):
    expected = identity_block()
    return {key: (value.get(key), expected[key]) for key in IDENTITY_KEYS
            if not isinstance(value, dict) or value.get(key) != expected[key]}


class LegacySchemaError(ValueError):
    """A schema <= 6 (or unlabelled) checkpoint was offered to the r2 reader."""


class ForeignContainerError(ValueError):
    """A smoke/initialization/r1 container was offered as a pilot/formal resume."""


def reject_foreign_container(state):
    """Reject non-resume containers before any schema or tensor inspection.

    SDS 7.5: the r1 full checkpoint, the r1 initialization container
    (``scope='initialization'``) and the old/ r2 smoke containers are never
    resume sources.  The smoke marker is detected from the container identity
    even when its ``schema`` field was edited to look newer.
    """
    if not isinstance(state, dict):
        raise ValueError('Checkpoint must be a mapping')
    for marker in SMOKE_CONTAINER_IDS:
        if marker in state:
            raise ForeignContainerError(
                f'smoke container {marker} is not a pilot/formal resume source')
        for value in state.values():
            if isinstance(value, str) and value == marker:
                raise ForeignContainerError(
                    f'smoke container {marker} is not a pilot/formal resume source')
    if state.get('container') in SMOKE_CONTAINER_IDS:
        raise ForeignContainerError('smoke container is not a pilot/formal resume source')
    if state.get('scope') == INITIALIZATION_SCOPE:
        raise ForeignContainerError(
            'initialization container (scope=initialization) is migration evidence, '
            'not a resume checkpoint')
    if state.get('formal_resumable') is False:
        raise ForeignContainerError('formal_resumable=false containers cannot be resumed')
    return state


def validate_checkpoint(state, *, expected_contract_sha256=None, model_factory=None,
                        parameters=None, cfg=None):
    """Strict schema-7 validation in fail-closed order; returns the state.

    Identity, schema, recipe and contract gates run before any tensor body is
    inspected, so a legacy/foreign envelope is rejected without needing the
    current model implementation.  ``model_factory`` (default
    ``v05.model.V05ActorCritic``) and ``parameters`` are only required for the
    model/Adam body checks.
    """
    reject_foreign_container(state)
    schema = state.get('schema')
    if schema is None:
        if 'contract' in state:
            raise LegacySchemaError('legacy checkpoint without an r2 schema marker '
                                    '(schema <= 6) is not resumable')
        raise LegacySchemaError('checkpoint has no schema marker; only schema 7 is accepted')
    if isinstance(schema, bool) or not isinstance(schema, int) or schema != SCHEMA_VERSION:
        if not isinstance(schema, int) or isinstance(schema, bool) or schema <= 6:
            raise LegacySchemaError(f'legacy schema {schema!r} (<= 6) is not resumable')
        raise ValueError(f'unknown checkpoint schema {schema!r}')
    if set(state) != CHECKPOINT_KEYS:
        missing = sorted(CHECKPOINT_KEYS - set(state))
        extra = sorted(set(state) - CHECKPOINT_KEYS)
        raise ValueError(f'schema7 field contract mismatch; missing={missing} extra={extra}')
    validate_observation_contract()
    mismatch = _identity_mismatch(state['identity'])
    if mismatch:
        raise ValueError(f'checkpoint identity mismatch: {mismatch}')
    if state['recipe'] != recipe_config():
        raise ValueError('checkpoint recipe differs from the frozen 0.5-r2 recipe')
    if state['resume_mode'] != RESUME_MODE:
        raise ValueError('schema7 resume_mode must be new_episode')
    source = state['source']
    if not isinstance(source, dict) or set(source) != SOURCE_KEYS:
        raise ValueError('checkpoint source schema mismatch')
    source_block(**{key: source[key] for key in ('mode', 'seed', 'device')})
    if not isinstance(source['init_path'], str) or not os.path.isabs(source['init_path']):
        raise ValueError('initial checkpoint path must be absolute')
    if source['init_sha256'] != INIT_SHA256:
        raise ValueError('initial checkpoint SHA differs from the approved ETH source')
    if not isinstance(source['resume'], list):
        raise ValueError('resume lineage must be a list')
    require_hex(state['contract_sha256'], 'contract_sha256')
    expected_contract = (expected_contract_sha256 if expected_contract_sha256 is not None
                         else contract_sha_for(source))
    if state['contract_sha256'] != expected_contract:
        raise ValueError('checkpoint contract/source closure differs from this v0.5 build')
    update = validate_learning(state['learning'])
    execution_mode = source['mode']
    validate_curriculum(state['curriculum'], learning=state['learning'],
                        mode=execution_mode)
    audit_budget(state['learning'], state['curriculum'], mode=execution_mode)
    task_state = state['ledger']
    if not isinstance(task_state, dict) or set(task_state) != LEDGER_STATE_KEYS:
        raise ValueError('checkpoint ledger task-state schema mismatch')
    validate_task_state(task_state, curriculum=state['curriculum'])
    evaluation = state['evaluation']
    if not isinstance(evaluation, dict) or set(evaluation) != EVALUATION_KEYS:
        raise ValueError('checkpoint evaluation schema mismatch')
    consumed = evaluation['consumed']
    if not isinstance(consumed, list) or len(set(consumed)) != len(consumed):
        raise ValueError('evaluation consumed keys must be a unique list')
    for key in consumed:
        record = task_state['tasks'].get(key)
        if not isinstance(record, dict) or record.get('state') != 'consumed':
            raise ValueError('evaluation consumed key has no consumed task record')
    if not isinstance(state['model'], dict) or not state['model']:
        raise ValueError('checkpoint model state required')
    validate_route_model_state(state['model'])
    if not isinstance(state['normalizer'], dict):
        raise ValueError('checkpoint normalizer state required')
    validate_rng(state['rng'])
    if model_factory is not None or parameters is not None:
        factory = model_factory if model_factory is not None else _default_model_factory
        reference = factory() if model_factory is not None else None
        params = list(parameters) if parameters is not None else list(reference.parameters())
        if model_factory is not None:
            _validate_model_body(reference, state['model'])
            params = list(reference.parameters())
            del reference
        validate_optimizer_state(state['optimizer'], params, state['learning'], cfg=cfg)
    if update != sum(state['curriculum']['spent'].values()):
        raise ValueError('curriculum spent differs from the learning boundary')
    return state


def validate_route_model_state(model_state):
    """SDS 7.5: reject a schema-7 body without the r2 route bypass / 211 Critic."""
    if not isinstance(model_state, dict):
        raise ValueError('checkpoint model state must be a state_dict mapping')
    for name, shape in ROUTE_MODEL_KEYS.items():
        if name not in model_state:
            raise ValueError(f'schema7 model is missing the r2 route tensor {name!r}')
        value = model_state[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f'schema7 model tensor {name!r} must be a tensor')
        if tuple(value.shape) != tuple(shape):
            raise ValueError(f'schema7 model tensor {name!r} must be {tuple(shape)}, '
                             f'got {tuple(value.shape)}')
        if not torch.isfinite(value).all():
            raise ValueError(f'schema7 model tensor {name!r} contains NaN/Inf')
    return model_state


def _default_model_factory():
    from .model import V05R2ActorCritic
    return V05R2ActorCritic()


def _validate_model_body(model, model_state):
    from .model import validate_model_state
    parameters = inspect.signature(validate_model_state).parameters
    positional = [name for name, item in parameters.items()
                  if item.kind in (item.POSITIONAL_ONLY, item.POSITIONAL_OR_KEYWORD)]
    if len(positional) >= 2:
        return validate_model_state(model_state, model)
    return validate_model_state(model_state)


def preload_checkpoint(path, expected_sha=None, *, execution=None, mode=None,
                       expected_contract_sha256=None, model_factory=None,
                       validate_bodies=False):
    """CPU-only schema-7 preload with an exact source SHA and strict identity.

    The envelope (schema/identity/recipe/contract/learning/curriculum/task
    ledger/RNG/source) is always validated.  Model and Adam *bodies* need the
    caller's model, so they are validated only when ``validate_bodies=True``
    (or a ``model_factory`` is supplied); ``v05_r2/runner.py`` resumes by calling
    ``validate_checkpoint(state, parameters=list(model.parameters()))`` right
    after this preload, which is the SDS 7.2 body check.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if expected_sha is None:
        raise ValueError('A trusted checkpoint SHA256 is required before loading')
    require_hex(expected_sha, 'expected checkpoint SHA256')
    if actual != expected_sha:
        raise ValueError('Checkpoint SHA mismatch')
    state = torch.load(path, map_location='cpu', weights_only=False)
    if mode is not None and isinstance(state, dict) and isinstance(state.get('source'), dict):
        if state['source'].get('mode') != mode:
            raise ValueError('checkpoint mode differs from the requested execution')
    if execution is not None:
        if isinstance(state, dict) and isinstance(state.get('source'), dict):
            if state['source'].get('mode') != execution:
                raise ValueError('checkpoint execution differs from the requested execution')
    if (validate_bodies or model_factory is not None) and model_factory is None:
        model_factory = _default_model_factory
    validate_checkpoint(state, expected_contract_sha256=expected_contract_sha256,
                        model_factory=model_factory)
    return state


# ------------------------------------------------------------------ task state
def task_key(*, run_id, mode, stage, global_update, kind, checkpoint_sha256, case_sha256):
    """Canonical unique evaluation task key (SDS 7.3 / CONTRACT 6)."""
    if not isinstance(run_id, str) or not run_id or '|' in run_id or run_id.strip() != run_id:
        raise ValueError('run_id must be a non-empty pipe-free string')
    if mode not in CONTRACT_MODES:
        raise ValueError(f'mode must be one of {CONTRACT_MODES}')
    if stage not in STAGE_BY_ID:
        raise ValueError('task stage must be one of the ten v0.5-r2 stages')
    integer(global_update, 'global_update')
    if kind not in EVALUATION_KINDS:
        raise ValueError(f'kind must be one of {EVALUATION_KINDS}')
    require_hex(checkpoint_sha256, 'checkpoint_sha256')
    require_hex(case_sha256, 'case_sha256')
    return '|'.join((run_id, mode, stage, str(global_update), kind,
                     checkpoint_sha256, case_sha256))


def split_task_key(key):
    parts = key.split('|')
    if len(parts) != len(TASK_KEY_FIELDS):
        raise ValueError('Malformed evaluation task key')
    values = dict(zip(TASK_KEY_FIELDS, parts))
    values['global_update'] = int(values['global_update'])
    rebuilt = task_key(**values)
    if rebuilt != key:
        raise ValueError('Evaluation task key is not canonical')
    return values


def validate_task_state(task_state, *, curriculum=None):
    if not isinstance(task_state, dict) or set(task_state) != LEDGER_STATE_KEYS:
        raise ValueError('ledger task-state schema mismatch')
    anchor = task_state['anchor']
    if (not isinstance(anchor, dict) or set(anchor) != {'seq', 'digest'}
            or isinstance(anchor['seq'], bool) or not isinstance(anchor['seq'], int)
            or anchor['seq'] < 0):
        raise ValueError('invalid ledger anchor')
    require_hex(anchor['digest'], 'ledger anchor digest')
    tasks = task_state['tasks']
    if not isinstance(tasks, dict):
        raise ValueError('ledger tasks must be a mapping')
    for key, record in tasks.items():
        split_task_key(key)
        if not isinstance(record, dict) or record.get('state') not in EVALUATION_STATES:
            raise ValueError('unknown evaluation task state')
        required = set(TASK_RECORD_KEYS)
        if record['state'] in ('result_committed', 'consumed'):
            required.add('result')
        if record['state'] == 'consumed':
            required.add('consumption')
        if set(record) != required:
            raise ValueError('evaluation task record schema mismatch')
        if record['kind'] != split_task_key(key)['kind']:
            raise ValueError('task record/kind mismatch')
        if 'result' in required:
            if not isinstance(record['result'], dict) or set(record['result']) != RESULT_KEYS:
                raise ValueError('evaluation result record schema mismatch')
        if 'consumption' in required:
            consumption = record['consumption']
            if (not isinstance(consumption, dict) or set(consumption) != CONSUMPTION_KEYS
                    or not isinstance(consumption['streak'], dict)
                    or set(consumption['streak']) != CURRICULUM_STREAK_KEYS):
                raise ValueError('evaluation consumption record schema mismatch')
    results = task_state['results']
    if not isinstance(results, list):
        raise ValueError('ledger results must be a list')
    for row in results:
        if not isinstance(row, dict) or set(row) != RESULT_RECORD_KEYS:
            raise ValueError('evaluation result record schema mismatch')
    integer(task_state['collapse_streak'], 'collapse streak')
    integer(task_state['gate_streak'], 'gate streak')
    if task_state['qualified_update'] is not None:
        integer(task_state['qualified_update'], 'qualified update')
    if curriculum is not None:
        if task_state['gate_streak'] != curriculum['streak']['gate']:
            raise ValueError('ledger gate streak differs from curriculum streak')
        if task_state['collapse_streak'] != curriculum['streak']['collapse']:
            raise ValueError('ledger collapse streak differs from curriculum streak')
        if task_state['pending_transition'] != curriculum['pending_transition']:
            raise ValueError('ledger pending transition differs from curriculum')
    return task_state


def next_streak_state(curriculum, record, *, passed, collapse_hit, mode='formal'):
    """Derive ``(streak, pending_transition)`` for one consumed evaluation.

    This is the single implementation of the SDS 6.1/6.2 streak rules: the
    ledger event and the caller must agree with it exactly, so a consumption can
    never change a streak that its evidence does not justify.
    """
    streak = dict(curriculum['streak'])
    pending = curriculum['pending_transition']
    kind = record['kind']
    if kind == 'validation':
        # Two consecutive passes qualify; the counter saturates at 2 so the pilot
        # P1 (which keeps running to its fixed budget) cannot inflate it.
        streak['gate'] = min(2, streak['gate'] + 1) if passed else 0
        # SDS 7.1: the pilot's P1 runs a fixed 500 updates and may not advance
        # early, so a P1 pass only advances the gate streak/result record.  F0 ->
        # P1 still advances, and formal keeps the two-pass transition.
        if streak['gate'] >= 2 and not (mode == 'pilot' and record['stage'] == 'P1'):
            index = STAGE_ORDER.index(record['stage'])
            target = STAGE_ORDER[index + 1] if index + 1 < len(STAGE_ORDER) else None
            pending = dict(from_stage=record['stage'], to_stage=target,
                           at_update=record['global_update'],
                           requires='perturbed' if record['stage'] == 'S1' else None,
                           at_checkpoint=record['checkpoint_sha256'])
        else:
            pending = None
    elif kind == 'perturbed':
        if passed:
            streak['gate'] = max(streak['gate'], 2)
            if pending is not None:
                pending = dict(pending, requires=None)
        else:
            streak['gate'] = 0
            pending = None
    elif kind == 'diagnostic':
        streak['collapse'] = streak['collapse'] + 1 if collapse_hit else 0
    elif kind in ('flat_retention', 'test'):
        pass
    else:  # pragma: no cover - kind is validated when the task is planned
        raise ValueError('Unknown evaluation kind')
    return streak, pending


class RecoveryBlocked(RuntimeError):
    """Fail-closed recovery: the ledger cannot price this resume."""

    def __init__(self, reason, *, audit=None):
        self.audit = dict(status='recovery_blocked', reason=reason, **(audit or {}))
        super().__init__(f'recovery_blocked: {reason}')


# ---------------------------------------------------------------- the ledger
class AttemptLedger:
    """Single-writer hash-chained JSONL ledger; every append is fsynced.

    The ledger owns the derived curriculum, streaks and evaluation task states:
    a consumed evaluation changes the streak inside the same event, and a stage
    advance exists only as one ``transition_committed`` event.  Nothing can be
    refunded: ``learning`` only moves forward by complete 48-control/20-Adam
    updates.
    """
    def __init__(self, directory, *, run_id, mode='formal', retain_events=RETAINED_EVENTS):
        if isinstance(retain_events, bool) or not isinstance(retain_events, int) \
                or retain_events < 1:
            raise ValueError('retain_events must be a positive integer')
        self.retain_events = retain_events
        self.path = Path(directory) / 'ledger.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        header = dict(schema=1, run_id=run_id, mode=mode)
        self._validate_header(header)
        if not self.path.exists():
            try:
                with open(self.path, 'xb') as stream:
                    stream.write((canonical({'header': header}) + '\n').encode())
                    stream.flush()
                    os.fsync(stream.fileno())
                fd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            except FileExistsError:
                pass
        raw = self.path.read_bytes()
        if not raw.endswith(b'\n'):
            raise ValueError('Truncated durable ledger: unknown physical/optimizer tail')
        try:
            lines = raw.decode().splitlines()
            stored = json.loads(lines[0])
            if stored != {'header': header}:
                raise ValueError('Ledger run/mode identity mismatch')
            events = [json.loads(line) for line in lines[1:]]
        except (UnicodeError, json.JSONDecodeError, IndexError) as error:
            raise ValueError('Damaged ledger; recovery blocked') from error
        self._initialize(header, events, retain=self.retain_events)
        self._disk_size = len(raw)

    # ---------------------------------------------------------- construction
    @staticmethod
    def _validate_header(header):
        if (not isinstance(header, dict) or set(header) != LEDGER_HEADER_KEYS
                or header['schema'] != 1 or not isinstance(header['run_id'], str)
                or not header['run_id'] or header['mode'] not in CONTRACT_MODES):
            raise ValueError('Invalid ledger identity')

    def _initialize(self, header, events, *, retain=None):
        from collections import deque
        self._validate_header(header)
        self.header = deepcopy(header)
        self.events = deque(maxlen=retain)
        self._event_count = 0
        self._digest = digest(header)
        self._data = dict(completed=0, control_samples=0, sampled=0, physics=0, discarded=0,
                          attempts=[], active=None, unknown=False, checkpoints=[],
                          curriculum=genesis_curriculum(), tasks={}, results=[],
                          qualified_update=None, mode=header['mode'])
        self._live_attempt_id = None
        self._collect_issued = None
        self._poisoned = False
        if not isinstance(events, list):
            raise ValueError('Missing ledger event history')
        for event in events:
            self._validate_event(event)
            self._apply(self._data, event)
            self.events.append(event)
            self._digest = event['digest']
            self._event_count += 1

    @classmethod
    def from_state(cls, state):
        if not isinstance(state, dict) or set(state) != LEDGER_SNAPSHOT_KEYS:
            raise ValueError('Ledger snapshot schema mismatch')
        result = object.__new__(cls)
        result.path = None
        result.retain_events = len(state['events']) or 1
        result._initialize(state['header'], state['events'])
        result._disk_size = None
        if result.anchor()['seq'] != state['seq'] or result._digest != state['digest']:
            raise ValueError('Ledger snapshot anchor mismatch')
        return result

    @classmethod
    def replay_prefix(cls, header, events, seq):
        """Derived state after exactly ``seq`` events (used for checkpoint audit)."""
        result = object.__new__(cls)
        result.path = None
        result.retain_events = max(1, len(events))
        result._initialize(header, list(events)[:seq])
        result._disk_size = None
        if seq and result.anchor()['digest'] != events[seq - 1]['digest']:
            raise ValueError('Ledger prefix digest mismatch')
        return result

    # ------------------------------------------------------------ validation
    def _validate_event(self, event):
        if (not isinstance(event, dict) or set(event) != {'seq', 'previous', 'event',
                                                          'payload', 'digest'}
                or isinstance(event['seq'], bool) or not isinstance(event['seq'], int)
                or event['seq'] != self._event_count + 1
                or event['previous'] != self._digest
                or event['digest'] != digest({key: value for key, value in event.items()
                                              if key != 'digest'})):
            raise ValueError('Ledger chain damaged or reordered')

    @staticmethod
    def _curriculum_progress(curriculum):
        return deepcopy(curriculum)

    def _expected_learning(self, update, optimizer_step=None):
        learning = learning_state(update, num_envs=self._env_count())
        if optimizer_step is not None:
            learning['optimizer_step'] = optimizer_step
        return learning

    def _env_count(self):
        segments = self._data.get('attempts', [])
        if self._data['active'] is not None:
            return self._data['active']['num_envs']
        return segments[-1]['num_envs'] if segments else 0

    @staticmethod
    def _apply(data, event):
        kind, payload, active = event['event'], event['payload'], data['active']
        if not isinstance(payload, dict):
            raise ValueError('Invalid ledger payload')
        curriculum = data['curriculum']

        def expected_learning(update, env_count, optimizer_step=None):
            learning = learning_state(update, num_envs=env_count)
            if optimizer_step is not None:
                learning['optimizer_step'] = optimizer_step
            return learning

        if kind == 'begin_attempt':
            if set(payload) != {'attempt', 'before', 'num_envs'} or active is not None or data['unknown']:
                raise ValueError('Cannot begin overlapping/unknown attempt')
            env_count = payload['num_envs']
            before = payload['before']
            if (isinstance(env_count, bool) or not isinstance(env_count, int)
                    or not 1 <= env_count <= 4096 or isinstance(payload['attempt'], bool)
                    or not isinstance(payload['attempt'], int)
                    or payload['attempt'] != len(data['attempts']) + 1):
                raise ValueError('Attempt does not match complete learning progress')
            validate_curriculum(before, learning=learning_state(data['completed'],
                                                               num_envs=env_count),
                                mode=data['mode'])
            if before != curriculum:
                raise ValueError('Attempt curriculum differs from the durable curriculum')
            item = dict(attempt=payload['attempt'], before=deepcopy(before), num_envs=env_count,
                        phase='collect', controls=0, optimizer_steps=0, core_completed=False)
            data['active'] = item
        elif kind == 'control_step':
            if (set(payload) != {'attempt', 'step_index', 'num_envs',
                                 'control_samples_increment'}
                    or active is None or active['phase'] != 'collect'
                    or payload['attempt'] != active['attempt']
                    or isinstance(payload['step_index'], bool)
                    or not isinstance(payload['step_index'], int)
                    or payload['step_index'] != active['controls'] + 1
                    or payload['step_index'] > PPO_ROLLOUT
                    or payload['num_envs'] != active['num_envs']
                    # The legacy callback name is a control-sample count: mapped
                    # once here, never multiplied by the decimation again.
                    or payload['control_samples_increment'] != active['num_envs']):
                raise ValueError('Control samples missing, duplicate or mismatched')
            active['controls'] += 1
            # Observed control samples (durable per-step evidence) are kept apart
            # from the control samples owned by complete PPO updates.
            data['sampled'] += active['num_envs']
            data['physics'] += DECIMATION * active['num_envs']
        elif kind == 'optimizer_started':
            if (set(payload) != {'attempt', 'learning_state', 'expected_optimizer_steps'}
                    or active is None or active['phase'] != 'collect' or active['controls'] != PPO_ROLLOUT
                    or payload['attempt'] != active['attempt']
                    or payload['expected_optimizer_steps'] != ADAM_STEPS_PER_UPDATE
                    or payload['learning_state'] != expected_learning(data['completed'],
                                                                     active['num_envs'])):
                raise ValueError('Optimizer start outside a complete rollout boundary')
            active['phase'] = 'optimizer_started'
        elif kind == 'optimizer_step_completed':
            if (set(payload) != {'attempt', 'learning_state'} or active is None
                    or active['phase'] != 'optimizer_started'
                    or payload['attempt'] != active['attempt']
                    or active['optimizer_steps'] >= ADAM_STEPS_PER_UPDATE):
                raise ValueError('Unexpected Adam callback')
            expected = expected_learning(
                data['completed'], active['num_envs'],
                optimizer_step=ADAM_STEPS_PER_UPDATE * data['completed']
                + active['optimizer_steps'] + 1)
            if payload['learning_state'] != expected:
                raise ValueError('Adam callback step identity mismatch')
            active['optimizer_steps'] += 1
        elif kind == 'update_completed':
            if (set(payload) != {'attempt', 'learning_state', 'optimizer_steps_in_update'}
                    or active is None or active['phase'] != 'optimizer_started'
                    or active['optimizer_steps'] != ADAM_STEPS_PER_UPDATE or active['core_completed']
                    or payload['attempt'] != active['attempt']
                    or payload['optimizer_steps_in_update'] != ADAM_STEPS_PER_UPDATE
                    or payload['learning_state'] != expected_learning(data['completed'] + 1,
                                                                     active['num_envs'])):
                raise ValueError('Core completion without exactly 20 Adam steps')
            active['core_completed'] = True
            active['phase'] = 'complete_pending'
        elif kind == 'complete_attempt':
            if (set(payload) != {'attempt', 'curriculum', 'learning_state'}
                    or active is None or active['phase'] != 'complete_pending'
                    or payload['attempt'] != active['attempt']
                    or payload['learning_state'] != expected_learning(data['completed'] + 1,
                                                                     active['num_envs'])):
                raise ValueError('Complete attempt without the exact learning boundary')
            expected = updated_curriculum(active['before'], num_envs=active['num_envs'],
                                          mode=data['mode'])
            if payload['curriculum'] != expected:
                raise ValueError('Curriculum update missing, duplicated or inconsistent')
            active['phase'] = 'complete'
            data['completed'] += 1
            data['control_samples'] += PPO_ROLLOUT * active['num_envs']
            data['curriculum'] = deepcopy(expected)
            data['attempts'] = [*data['attempts'], active]
            data['active'] = None
        elif kind == 'discard_collect_only':
            if (set(payload) != {'attempt', 'known_no_inflight', 'proof'} or active is None
                    or active['phase'] != 'collect' or payload['attempt'] != active['attempt']
                    or payload['known_no_inflight'] is not True
                    or not isinstance(payload['proof'], str) or not payload['proof']):
                raise ValueError('Collect-only discard lacks positive evidence')
            active['phase'] = 'discarded'
            active['closure_proof'] = payload['proof']
            data['discarded'] += active['controls'] * active['num_envs']
            data['attempts'] = [*data['attempts'], active]
            data['active'] = None
        elif kind == 'unknown_tail':
            if set(payload) != {'reason'} or not isinstance(payload['reason'], str) or not payload['reason']:
                raise ValueError('Unknown tail reason required')
            data['unknown'] = True
        elif kind == 'checkpoint_committed':
            if (set(payload) != {'path', 'sha256', 'anchor', 'update', 'contract_sha256'}
                    or active is not None or data['unknown']
                    or not isinstance(payload['path'], str)
                    or not os.path.isabs(payload['path'])
                    or not HEX.fullmatch(str(payload['sha256']))
                    or not HEX.fullmatch(str(payload['contract_sha256']))
                    or payload['update'] != data['completed']
                    or payload['anchor'] != dict(seq=event['seq'] - 1, digest=event['previous'])):
                raise ValueError('Checkpoint commitment is not at a durable complete boundary')
            data['checkpoints'] = [*data['checkpoints'], deepcopy(payload)]
        elif kind == 'evaluation_planned':
            required = {'task_key', 'run_id', 'mode', 'stage', 'global_update', 'kind',
                        'checkpoint_sha256', 'case_sha256', 'planned_wall'}
            if set(payload) != required or data['unknown']:
                raise ValueError('Evaluation plan schema mismatch')
            key = task_key(**{field: payload[field] for field in TASK_KEY_FIELDS})
            if key != payload['task_key']:
                raise ValueError('Evaluation plan key is not canonical')
            if payload['mode'] != data['mode']:
                raise ValueError('Evaluation mode differs from the ledger mode')
            if payload['global_update'] != data['completed']:
                raise ValueError('Evaluation must bind the current complete update boundary')
            if payload['kind'] == 'test' and data['qualified_update'] is None:
                raise ValueError('Test evaluation requires a qualified terminal stage')
            if key in data['tasks']:
                raise ValueError('Duplicate evaluation plan event for one key')
            record = {field: payload[field] for field in TASK_RECORD_KEYS if field != 'state'}
            record['state'] = 'planned'
            data['tasks'][key] = record
        elif kind == 'evaluation_result_committed':
            required = {'task_key', 'result_path', 'result_sha256', 'report_path',
                        'report_sha256', 'evidence_sha256', 'summary_sha256', 'committed_wall'}
            if set(payload) != required:
                raise ValueError('Evaluation result commitment schema mismatch')
            record = data['tasks'].get(payload['task_key'])
            if record is None or record['state'] != 'planned':
                raise ValueError('Result commitment without a planned evaluation task')
            for field in ('result_sha256', 'report_sha256', 'summary_sha256'):
                if not HEX.fullmatch(str(payload[field])):
                    raise ValueError(f'Invalid {field}')
            evidence = payload['evidence_sha256']
            if (not isinstance(evidence, list) or not evidence
                    or any(not HEX.fullmatch(str(item)) for item in evidence)):
                raise ValueError('Committed raw evidence SHA256 list required')
            for field in ('result_path', 'report_path'):
                if not isinstance(payload[field], str) or not os.path.isabs(payload[field]):
                    raise ValueError(f'{field} must be absolute')
            record['state'] = 'result_committed'
            record['result'] = {key: deepcopy(value) for key, value in payload.items()
                                if key != 'task_key'}
        elif kind == 'evaluation_consumed':
            required = {'task_key', 'result_sha256', 'outcome', 'passed', 'collapse_hit',
                        'result_path', 'report_path', 'report_sha256', 'summary_sha256',
                        'evidence_sha256', 'consumed_wall', 'streak_after'}
            if set(payload) != required:
                raise ValueError('Evaluation consumption schema mismatch')
            record = data['tasks'].get(payload['task_key'])
            if record is None or record['state'] != 'result_committed':
                raise ValueError('Consumption requires a committed result')
            result = record['result']
            for field in ('result_sha256', 'report_path', 'report_sha256', 'summary_sha256',
                          'evidence_sha256'):
                if payload[field] != result[field] and not (field == 'result_path'):
                    raise ValueError('Consumption differs from the committed result evidence')
            if payload['result_path'] != result['result_path']:
                raise ValueError('Consumption differs from the committed result path')
            if (not isinstance(payload['outcome'], str) or not payload['outcome']
                    or not isinstance(payload['passed'], bool)
                    or not isinstance(payload['collapse_hit'], bool)):
                raise ValueError('Consumption outcome flags must be booleans')
            kind_name = record['kind']
            streak, pending = next_streak_state(curriculum, record,
                                                passed=payload['passed'],
                                                collapse_hit=payload['collapse_hit'],
                                                mode=data['mode'])
            expected_streak = dict(gate=streak['gate'], collapse=streak['collapse'])
            if payload['streak_after'] != expected_streak:
                raise ValueError('Consumption streak change is not the ledger-derived one')
            new_curriculum = deepcopy(curriculum)
            new_curriculum['streak'] = expected_streak
            new_curriculum['pending_transition'] = pending
            validate_curriculum(new_curriculum,
                                learning=learning_state(data['completed'],
                                                       num_envs=_attempt_env_count(data)),
                                mode=data['mode'])
            record['state'] = 'consumed'
            record['consumption'] = {field: deepcopy(payload[field]) for field in CONSUMPTION_KEYS
                                     if field in payload}
            record['consumption'].update(outcome=payload['outcome'],
                                         passed=payload['passed'],
                                         collapse_hit=payload['collapse_hit'],
                                         result_sha256=payload['result_sha256'],
                                         consumed_wall=payload['consumed_wall'],
                                         streak=deepcopy(expected_streak))
            data['curriculum'] = new_curriculum
            data['results'] = [*data['results'], dict(
                task_key=payload['task_key'], kind=kind_name, stage=record['stage'],
                global_update=record['global_update'], outcome=payload['outcome'],
                passed=payload['passed'], gate_streak=expected_streak['gate'],
                collapse_streak=expected_streak['collapse'])]
            if (kind_name == 'validation' and expected_streak['gate'] >= 2
                    and pending is not None and pending['to_stage'] is None):
                data['qualified_update'] = record['global_update']
        elif kind == 'transition_committed':
            required = {'from_stage', 'to_stage', 'at_update', 'gate_streak'}
            if set(payload) != required or active is not None or data['unknown']:
                raise ValueError('Transition must be committed at a complete boundary')
            pending = curriculum['pending_transition']
            if pending is None:
                raise ValueError('No pending transition to commit')
            if (payload['from_stage'] != pending['from_stage']
                    or payload['to_stage'] != pending['to_stage']
                    or payload['at_update'] != pending['at_update']
                    or payload['gate_streak'] != curriculum['streak']['gate']):
                raise ValueError('Transition differs from the pending transition')
            if pending['to_stage'] is None:
                raise ValueError('No next stage exists; the terminal state is not a transition')
            if pending['requires'] == 'perturbed':
                satisfied = any(
                    row['kind'] == 'perturbed' and row['passed']
                    and row['stage'] == pending['from_stage']
                    and row['global_update'] == pending['at_update']
                    and data['tasks'][row['task_key']]['checkpoint_sha256'] == pending['at_checkpoint']
                    for row in data['results'])
                if not satisfied:
                    raise ValueError('S1 extra perturbed gate is not consumed for this checkpoint')
            data['curriculum'] = advance_curriculum(curriculum, mode=data['mode'])
        else:
            raise ValueError('Unknown ledger event')

    # ------------------------------------------------------------------ append
    def _append(self, kind, payload):
        if self.path is None or self._poisoned:
            raise ValueError('Non-writable or failed ledger owner')
        event = dict(seq=self._event_count + 1, previous=self._digest, event=kind,
                     payload=deepcopy(payload))
        event['digest'] = digest(event)
        data = deepcopy(self._data)
        self._apply(data, event)
        encoded = (canonical(event) + '\n').encode()
        try:
            with open(self.path, 'ab') as stream:
                fcntl.flock(stream, fcntl.LOCK_EX)
                if os.fstat(stream.fileno()).st_size != self._disk_size:
                    raise ValueError('Stale concurrent ledger writer')
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
                self._disk_size += len(encoded)
        except BaseException:
            self._poisoned = True
            raise
        self.events.append(event)
        self._digest = event['digest']
        self._event_count += 1
        self._data = data

    # -------------------------------------------------------------- properties
    @property
    def mode(self):
        return self.header['mode']

    @property
    def completed_ppo_updates(self):
        return self._data['completed']

    @property
    def control_samples(self):
        """Control samples owned by complete updates (the learning boundary)."""
        return self._data['control_samples']

    @property
    def sampled_control_samples(self):
        """Control samples acknowledged by durable per-step callbacks."""
        return self._data['sampled']

    @property
    def physics_substeps(self):
        return self._data['physics']

    @property
    def discarded_samples(self):
        return self._data['discarded']

    @property
    def physics_substeps_observed(self):
        return self._data['physics']

    @property
    def physics_samples_unknown(self):
        active = self._data['active']
        return self._data['unknown'] or bool(active and active['phase'] == 'collect'
                                             and active['controls'] < PPO_ROLLOUT)

    @property
    def update_attempts(self):
        return deepcopy(self._data['attempts'] + ([self._data['active']]
                                                  if self._data['active'] else []))

    @property
    def active_attempt(self):
        return deepcopy(self._data['active'])

    @property
    def curriculum(self):
        return deepcopy(self._data['curriculum'])

    @property
    def gate_streak(self):
        return self._data['curriculum']['streak']['gate']

    @property
    def collapse_streak(self):
        return self._data['curriculum']['streak']['collapse']

    @property
    def pending_transition(self):
        return deepcopy(self._data['curriculum']['pending_transition'])

    @property
    def qualified_update(self):
        return self._data['qualified_update']

    @property
    def checkpoints(self):
        return deepcopy(self._data['checkpoints'])

    @property
    def tasks(self):
        return deepcopy(self._data['tasks'])

    @property
    def results(self):
        return deepcopy(self._data['results'])

    def task_state(self, key):
        record = self._data['tasks'].get(key)
        return None if record is None else record['state']

    def state_dict(self):
        """Complete event history, or a hard error when it is truncated in memory."""
        if len(self.events) != self._event_count:
            raise ValueError('ledger history is truncated in memory; '
                             'read the durable ledger file instead')
        return dict(header=deepcopy(self.header), events=deepcopy(list(self.events)),
                    seq=self._event_count, digest=self._digest)

    def anchor(self):
        return dict(seq=self._event_count, digest=self._digest)

    def _durable_events(self, seq):
        """Read exactly ``seq`` payload-bearing events back from the JSONL file."""
        if self.path is None:
            raise ValueError('Ledger prefix is not resident and has no durable file')
        with open(self.path, 'rb') as stream:
            header = json.loads(stream.readline())
            if header != {'header': self.header}:
                raise ValueError('Ledger header changed while reading a prefix')
            events = []
            for _ in range(seq):
                line = stream.readline()
                if not line.endswith(b'\n'):
                    raise ValueError('Durable ledger prefix is truncated')
                events.append(json.loads(line))
        return events

    def snapshot_at(self, seq):
        """Derived state after exactly ``seq`` durable events.

        The newest boundary is a copy of the current state (O(1)).  An older
        checkpoint anchor is re-read from the durable JSONL file, so bounded
        memory never changes the audit result.
        """
        if isinstance(seq, bool) or not isinstance(seq, int) \
                or not 0 <= seq <= self._event_count:
            raise ValueError('Ledger prefix anchor out of range')
        if seq == self._event_count:
            from collections import deque
            result = object.__new__(AttemptLedger)
            result.path = None
            result.retain_events = self.retain_events
            result.header = deepcopy(self.header)
            result.events = deque(self.events)
            result._event_count = self._event_count
            result._digest = self._digest
            result._data = deepcopy(self._data)
            result._live_attempt_id = None
            result._collect_issued = None
            result._poisoned = True
            result._disk_size = None
            return result
        events = self._durable_events(seq)
        return AttemptLedger.replay_prefix(self.header, events, seq)

    # ------------------------------------------------------- curriculum checks
    def expected_learning(self):
        env_count = self._env_count()
        if self._data['completed'] == 0:
            return learning_state(0, num_envs=512)
        return learning_state(self._data['completed'], num_envs=env_count)

    def require_curriculum(self, curriculum):
        """The runner's curriculum must equal the ledger-derived curriculum."""
        validate_curriculum(curriculum, learning=self.expected_learning(), mode=self.mode)
        if curriculum != self._data['curriculum']:
            raise ValueError('Curriculum differs from the durable ledger-derived state')

    def record_curriculum(self, curriculum):
        """No-op consistency check: v0.5 has no free-standing curriculum commits."""
        self.require_curriculum(curriculum)
        return self.anchor()

    # --------------------------------------------------------- attempt events
    def begin_attempt(self, curriculum, num_envs):
        self.require_curriculum(curriculum)
        if self._data['curriculum']['pending_transition'] is not None:
            raise ValueError('Cannot attempt training across a pending stage transition')
        stage = self._data['curriculum']['stage']
        spent = self._data['curriculum']['spent'][stage]
        if (spent >= stage_budgets(self.mode)[stage]
                or sum(self._data['curriculum']['spent'].values()) >= total_budget(self.mode)):
            raise ValueError('Stage budget already exhausted; no further update may charge')
        self._append('begin_attempt', dict(attempt=len(self._data['attempts']) + 1,
                                           before=self._curriculum_progress(curriculum),
                                           num_envs=num_envs))
        self._live_attempt_id = self._data['active']['attempt']
        return self._live_attempt_id

    def require_collect(self, num_envs):
        active = self._data['active']
        if (active is None or active['phase'] != 'collect' or active['controls'] != 0
                or active['num_envs'] != num_envs or self._live_attempt_id != active['attempt']
                or self._collect_issued == active['attempt']):
            raise ValueError('Collect requires a fresh durable live attempt')
        self._collect_issued = active['attempt']

    def require_update(self, num_envs):
        active = self._data['active']
        if (active is None or active['phase'] != 'collect' or active['controls'] != PPO_ROLLOUT
                or active['num_envs'] != num_envs or self._live_attempt_id != active['attempt']):
            raise ValueError('Update requires this live complete 48-step rollout')

    def control_step(self, event):
        active = self._data['active']
        if active is None or self._live_attempt_id != active['attempt']:
            raise ValueError('No live physical attempt')
        # Legacy callback field name kept for v0.3 compatibility; it counts
        # control samples, so it is mapped once and never multiplied again.
        increment = event.get('physics_samples_increment', event.get('control_samples_increment'))
        self._append('control_step', dict(attempt=active['attempt'],
                                          step_index=event.get('step_index'),
                                          num_envs=event.get('num_envs'),
                                          control_samples_increment=increment))

    def update_event(self, event, payload):
        active = self._data['active']
        if active is None or self._live_attempt_id != active['attempt']:
            raise ValueError('No live optimizer attempt')
        keys = {'optimizer_started': ('learning_state', 'expected_optimizer_steps'),
                'optimizer_step_completed': ('learning_state',),
                'update_completed': ('learning_state', 'optimizer_steps_in_update')}
        if event not in keys:
            raise ValueError('Unsupported core callback')
        self._append(event, dict(attempt=active['attempt'],
                                 **{key: payload.get(key) for key in keys[event]}))

    def complete_attempt(self, curriculum, learning):
        active = self._data['active']
        if active is None or self._live_attempt_id != active['attempt']:
            raise ValueError('No live complete attempt')
        self._append('complete_attempt', dict(attempt=active['attempt'],
                                              curriculum=self._curriculum_progress(curriculum),
                                              learning_state=deepcopy(learning)))
        self._live_attempt_id = None

    def close_collect_only(self, *, known_no_inflight=False, proof=None):
        active = self._data['active']
        if active is None or active['phase'] != 'collect':
            raise ValueError('Only a pre-optimizer collect can be discarded')
        if active['controls'] == PPO_ROLLOUT:
            proof = proof or 'all 48 control callbacks durable; no optimizer_started'
            known_no_inflight = True
        elif not (self._live_attempt_id == active['attempt'] and known_no_inflight is True
                  and isinstance(proof, str) and proof):
            raise ValueError('Partial collect requires live synchronous no-inflight proof')
        self._append('discard_collect_only', dict(attempt=active['attempt'],
                                                  known_no_inflight=known_no_inflight,
                                                  proof=proof))
        self._live_attempt_id = None

    def mark_unknown(self, reason):
        self._append('unknown_tail', dict(reason=reason))

    def record_checkpoint(self, path, checkpoint_sha, anchor, contract_sha256_value):
        if anchor != self.anchor():
            raise ValueError('Checkpoint/ledger anchor changed during save')
        require_hex(checkpoint_sha, 'checkpoint SHA256')
        require_hex(contract_sha256_value, 'contract SHA256')
        self._append('checkpoint_committed',
                     dict(path=str(Path(path).resolve()), sha256=checkpoint_sha,
                          anchor=deepcopy(anchor), update=self._data['completed'],
                          contract_sha256=contract_sha256_value))

    # ------------------------------------------------- evaluation transactions
    def plan_evaluation(self, *, run_id, mode, stage, global_update, kind,
                        checkpoint_sha256, case_sha256, planned_wall=None):
        """Persist the evaluation intent before any evaluation work starts."""
        key = task_key(run_id=run_id, mode=mode, stage=stage, global_update=global_update,
                       kind=kind, checkpoint_sha256=checkpoint_sha256,
                       case_sha256=case_sha256)
        record = self._data['tasks'].get(key)
        if record is not None:
            return dict(task_key=key, state=record['state'], record=deepcopy(record),
                        replayed=True)
        self._append('evaluation_planned', dict(
            task_key=key, run_id=run_id, mode=mode, stage=stage,
            global_update=global_update, kind=kind, checkpoint_sha256=checkpoint_sha256,
            case_sha256=case_sha256, planned_wall=float(planned_wall or 0.)))
        return dict(task_key=key, state='planned', record=deepcopy(self._data['tasks'][key]),
                    replayed=False)

    def commit_evaluation_result(self, key, *, result_path, result_sha256, report_path,
                                 report_sha256, evidence_sha256, summary_sha256,
                                 committed_wall=None):
        """Atomically record raw evidence + report; refuses a missing artefact."""
        record = self._data['tasks'].get(key)
        if record is None:
            raise ValueError('Result commitment without a planned evaluation task')
        for label, path, expected in (('result', result_path, result_sha256),
                                      ('report', report_path, report_sha256)):
            require_hex(expected, f'{label}_sha256')
            if not Path(path).is_file():
                raise ValueError(f'{label} evidence is not atomically committed on disk yet')
            if sha256(path) != expected:
                raise ValueError(f'{label}_sha256 does not match the committed file')
        evidence_sha256 = list(evidence_sha256)
        if not evidence_sha256:
            raise ValueError('A committed raw-evidence SHA256 list is required')
        for item in evidence_sha256:
            require_hex(item, 'evidence SHA256')
        require_hex(summary_sha256, 'summary_sha256')
        claimed = dict(result_path=str(Path(result_path).resolve()),
                       result_sha256=result_sha256,
                       report_path=str(Path(report_path).resolve()),
                       report_sha256=report_sha256,
                       evidence_sha256=evidence_sha256,
                       summary_sha256=summary_sha256,
                       committed_wall=float(committed_wall or 0.))
        if record['state'] in ('result_committed', 'consumed'):
            # Replaying the identical commitment is idempotent (never a new event
            # and never a second consumption).
            if record['result'] != claimed:
                raise ValueError('Replayed result commitment differs from the durable one')
            return dict(task_key=key, state=record['state'], replayed=True)
        self._append('evaluation_result_committed', dict(task_key=key, **claimed))
        return dict(task_key=key, state='result_committed', replayed=False)

    def consume_evaluation(self, key, *, outcome, passed, collapse_hit, consumed_wall=None):
        """Consume a committed result and change streaks in the same event."""
        record = self._data['tasks'].get(key)
        if record is None:
            raise ValueError('Consumption without a planned evaluation task')
        if record['state'] == 'consumed':
            prior = record['consumption']
            if (prior['outcome'] != outcome or prior['passed'] != passed
                    or prior['collapse_hit'] != collapse_hit):
                raise ValueError('Replayed consumption differs from the durable one')
            return dict(task_key=key, state='consumed', replayed=True,
                        streak=deepcopy(self._data['curriculum']['streak']))
        if record['state'] != 'result_committed':
            raise ValueError('Consumption requires a committed result')
        result = record['result']
        streak, _pending = next_streak_state(self._data['curriculum'], record,
                                             passed=bool(passed),
                                             collapse_hit=bool(collapse_hit),
                                             mode=self.mode)
        # A missing/changed artefact can only be repaired by re-running the exact
        # same case on the exact same model; it is never skipped or double-counted.
        for label, path, expected in (('result', result['result_path'], result['result_sha256']),
                                      ('report', result['report_path'], result['report_sha256'])):
            if not Path(path).is_file():
                raise ValueError(f'{label} evidence is missing; re-run the same case/model')
            if sha256(path) != expected:
                raise ValueError(f'{label} evidence changed; re-run the same case/model')
        payload = dict(task_key=key, result_sha256=result['result_sha256'],
                       outcome=outcome, passed=bool(passed), collapse_hit=bool(collapse_hit),
                       result_path=result['result_path'],
                       report_path=result['report_path'],
                       report_sha256=result['report_sha256'],
                       summary_sha256=result['summary_sha256'],
                       evidence_sha256=list(result['evidence_sha256']),
                       consumed_wall=float(consumed_wall or 0.),
                       streak_after=dict(gate=streak['gate'], collapse=streak['collapse']))
        self._append('evaluation_consumed', payload)
        return dict(task_key=key, state='consumed', replayed=False,
                    streak=deepcopy(self._data['curriculum']['streak']))

    def commit_transition(self, *, from_stage, to_stage, at_update, gate_streak):
        pending = self._data['curriculum']['pending_transition']
        if pending is not None:
            if (pending['from_stage'] != from_stage or pending['to_stage'] != to_stage
                    or pending['at_update'] != at_update):
                raise ValueError('Transition does not match the pending transition')
            self._append('transition_committed', dict(from_stage=from_stage, to_stage=to_stage,
                                                      at_update=at_update,
                                                      gate_streak=gate_streak))
            return dict(committed=True, replayed=False, stage=self._data['curriculum']['stage'])
        if self._data['curriculum']['stage'] == to_stage:
            # The exact transition is already durable (idempotent replay).
            return dict(committed=False, replayed=True, stage=self._data['curriculum']['stage'])
        raise ValueError('No pending transition matches this commit')

    def pending_evaluations(self):
        return {key: deepcopy(record) for key, record in self._data['tasks'].items()
                if record['state'] in ('planned', 'result_committed')}

    def missing_results(self):
        """Tasks whose committed result/report artefacts disappeared from disk."""
        missing = []
        for key, record in self._data['tasks'].items():
            if record['state'] == 'planned':
                continue
            result = record['result']
            paths = [result['result_path'], result['report_path']]
            broken = any(not Path(item).is_file() for item in paths)
            if not broken:
                broken = (sha256(result['result_path']) != result['result_sha256']
                          or sha256(result['report_path']) != result['report_sha256'])
            if broken:
                missing.append(dict(task_key=key, state=record['state'], kind=record['kind'],
                                    stage=record['stage'],
                                    global_update=record['global_update'],
                                    checkpoint_sha256=record['checkpoint_sha256'],
                                    case_sha256=record['case_sha256'],
                                    action='rerun_same_case_same_model'))
        return missing

    def checkpoint_task_state(self):
        return dict(tasks=deepcopy(self._data['tasks']),
                    results=deepcopy(self._data['results']),
                    collapse_streak=self._data['curriculum']['streak']['collapse'],
                    gate_streak=self._data['curriculum']['streak']['gate'],
                    anchor=self.anchor(),
                    qualified_update=self._data['qualified_update'],
                    pending_transition=deepcopy(self._data['curriculum']['pending_transition']))

    def checkpoint_evaluation(self):
        consumed = [key for key, record in self._data['tasks'].items()
                    if record['state'] == 'consumed']
        diagnostic = [record['global_update'] for record in self._data['tasks'].values()
                      if record['state'] == 'consumed' and record['kind'] == 'diagnostic']
        return dict(consumed=sorted(consumed),
                    latest_diagnostic_update=max(diagnostic) if diagnostic else None)


def _attempt_env_count(data):
    """Environment count of the last completed attempt (512 before the first)."""
    attempts = data['attempts']
    return attempts[-1]['num_envs'] if attempts else 512


# --------------------------------------------------------- newest checkpoint
def latest_checkpoint_record(ledger):
    """Newest verified complete-boundary checkpoint in ledger order (never mtime)."""
    records = ledger.checkpoints
    if not records:
        raise RecoveryBlocked('no complete checkpoint is recorded in the durable ledger')
    return max(records, key=lambda item: item['anchor']['seq'])


def same_update_records(ledger, update=None):
    records = ledger.checkpoints
    if not records:
        return []
    target = ledger.completed_ppo_updates if update is None else update
    return [record for record in records if record['update'] == target]


def select_checkpoint(ledger, *, directory=None, load=None):
    """Select the newest complete checkpoint for the exact saved boundary.

    Selection is by the ledger's durable complete-boundary order.  File mtimes
    are never consulted; a file that is not recorded by the ledger can never be
    chosen.  Two durable records with the same newest seq cannot happen (each
    event has a unique seq), so an unreadable newest record is a hard block
    instead of a silent fallback to an older same-update file.
    """
    record = latest_checkpoint_record(ledger)
    path = Path(record['path'])
    if directory is not None:
        path = Path(directory) / path.name
    if not path.is_file():
        raise RecoveryBlocked('newest complete checkpoint file is missing; '
                              'falling back to an older same-update state would roll back')
    if sha256(path) != record['sha256']:
        raise RecoveryBlocked('newest complete checkpoint bytes differ from the ledger record')
    if load is not None:
        state = load(path, record['sha256'])
        if state.get('ledger', {}).get('anchor') != record['anchor']:
            raise RecoveryBlocked('checkpoint ledger anchor differs from its durable record')
    return dict(path=path, sha256=record['sha256'], update=record['update'],
                anchor=record['anchor'], contract_sha256=record['contract_sha256'])


# --------------------------------------------------------------- audit/resume
def audit_recovery(checkpoint, ledger, *, discard_collect_only=False, selected=None):
    """No replay/refund: only an exact complete boundary plus proven collect tails."""
    details = dict(checkpoint_update=checkpoint.get('learning', {}).get('update'),
                   ledger_completed=ledger.completed_ppo_updates,
                   control_samples_observed_lower_bound=ledger.sampled_control_samples,
                   physics_substeps_observed_lower_bound=ledger.physics_substeps_observed,
                   physics_samples_unknown=ledger.physics_samples_unknown,
                   new_episode=True, physical_trajectory_continuous=False,
                   reward_state=REWARD_STATE, half_trajectory_restored=False)

    def blocked(reason):
        raise RecoveryBlocked(reason, audit=details)

    schema = checkpoint.get('schema')
    try:
        reject_foreign_container(checkpoint)
        validate_route_model_state(checkpoint.get('model'))
    except (ForeignContainerError, ValueError) as error:
        blocked(str(error))
    if schema != SCHEMA_VERSION:
        blocked(f'legacy or foreign checkpoint schema {schema!r}; only schema 7 resumes')
    if checkpoint.get('resume_mode') != RESUME_MODE:
        blocked('schema7 new-episode resume mode mismatch')
    try:
        mode = checkpoint.get('source', {}).get('mode')
        update = validate_learning(checkpoint['learning'])
        validate_curriculum(checkpoint['curriculum'], learning=checkpoint['learning'],
                            mode=mode)
        audit_budget(checkpoint['learning'], checkpoint['curriculum'], mode=mode)
        task_state = validate_task_state(checkpoint['ledger'])
    except (ValueError, KeyError, TypeError) as error:
        blocked(f'invalid checkpoint ledger: {error}')
    anchor = task_state['anchor']
    at_anchor = ledger.snapshot_at(anchor['seq'])
    if at_anchor.anchor()['digest'] != anchor['digest']:
        blocked('checkpoint ledger anchor is not a durable ledger prefix')
    if at_anchor.curriculum != checkpoint['curriculum']:
        blocked('checkpoint curriculum differs from the ledger at its anchor')
    if at_anchor.checkpoint_task_state() != checkpoint['ledger']:
        blocked('checkpoint task ledger differs from the ledger at its anchor')
    if at_anchor.checkpoint_evaluation() != checkpoint['evaluation']:
        blocked('checkpoint evaluation state differs from the ledger at its anchor')
    if update != at_anchor.completed_ppo_updates:
        blocked('checkpoint learning boundary differs from the ledger at its anchor')
    if ledger.completed_ppo_updates != update:
        blocked('unsaved complete PPO updates; rollback and refund are forbidden')
    newer = [record for record in ledger.checkpoints
             if record['anchor']['seq'] > anchor['seq']]
    if newer:
        if selected is None or selected['sha256'] != max(
                newer, key=lambda item: item['anchor']['seq'])['sha256']:
            blocked('newer complete checkpoint available; the older same-update state '
                    'must not be resumed')
    active = ledger.active_attempt
    if active is not None and active['phase'] != 'collect':
        blocked('optimizer_started or an incomplete Adam/learning commit')
    if ledger.physics_samples_unknown:
        blocked('unknown physical sample tail')
    discarded = ledger.discarded_samples - at_anchor.discarded_samples
    if active is not None:
        if active['controls'] != PPO_ROLLOUT:
            blocked('unknown partial collect tail')
        discarded += active['num_envs'] * PPO_ROLLOUT
        if discard_collect_only:
            ledger.close_collect_only()
    if ledger.sampled_control_samples != at_anchor.sampled_control_samples + discarded:
        blocked('control samples are not explained by complete PPO plus discarded collect-only')
    if ledger.control_samples != checkpoint['learning']['control_samples']:
        blocked('observed complete control samples differ from the checkpoint learning block')
    if ledger.physics_substeps_observed != DECIMATION * ledger.sampled_control_samples:
        blocked('physics_substeps is not exactly 4x control_samples (no double multiply)')
    pending = ledger.pending_evaluations()
    return dict(status='eligible', discarded_samples=discarded, **details,
                resume_completed_ppo_updates=update, resampling=discarded > 0,
                ledger_anchor=ledger.anchor(),
                pending_evaluations={key: record['state'] for key, record in pending.items()})


def resume_plan(ledger):
    """What must happen before PPO continues (SDS 7.3 evaluation transactions)."""
    rerun, consume, missing = [], [], []
    missing_keys = {item['task_key'] for item in ledger.missing_results()}
    for key, record in ledger.pending_evaluations().items():
        entry = dict(task_key=key, kind=record['kind'], stage=record['stage'],
                     global_update=record['global_update'],
                     checkpoint_sha256=record['checkpoint_sha256'],
                     case_sha256=record['case_sha256'], state=record['state'])
        if record['state'] == 'planned':
            rerun.append(entry)
        elif key in missing_keys:
            missing.append(dict(entry, action='rerun_same_case_same_model'))
        else:
            consume.append(entry)
    return dict(rerun=rerun, consume=consume, missing=missing,
                blocked_until_complete=bool(rerun or consume or missing),
                pending_transition=ledger.pending_transition,
                streak=dict(gate=ledger.gate_streak, collapse=ledger.collapse_streak))


# ------------------------------------------------------- state reconciliation
STATE_KEYS = frozenset({'schema', 'run_id', 'learning', 'curriculum', 'checkpoint',
                        'ledger_anchor', 'evaluation'})
STATE_CHECKPOINT_KEYS = frozenset({'path', 'sha256', 'update', 'ledger_anchor'})


def atomic_json(path, value):
    """fsync-backed atomic JSON (parent directory fsynced after replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp',
                                         dir=path.parent)
    os.close(handle)
    try:
        with open(temporary, 'w') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False,
                      allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_torch_save(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    os.close(handle)
    try:
        torch.save(state, temporary)
        with open(temporary, 'rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_state(path, mode='formal'):
    path = Path(path)
    if not path.is_file():
        return None
    state = json.loads(path.read_text())
    if not isinstance(state, dict) or set(state) != STATE_KEYS:
        raise ValueError('run state schema mismatch')
    if state['schema'] != SCHEMA_VERSION:
        raise LegacySchemaError('run state schema is not v0.5-r2 schema 7')
    validate_learning(state['learning'])
    validate_curriculum(state['curriculum'], learning=state['learning'], mode=mode)
    if not isinstance(state['checkpoint'], dict) or set(state['checkpoint']) != STATE_CHECKPOINT_KEYS:
        raise ValueError('run state checkpoint reference schema mismatch')
    if not isinstance(state['evaluation'], dict) or set(state['evaluation']) != EVALUATION_KEYS:
        raise ValueError('run state evaluation schema mismatch')
    return state


def _reachable(older, newer):
    """Can ``newer`` be the durable curriculum for the same model as ``older``?"""
    if set(older['spent']) != set(newer['spent']):
        return False
    for stage, value in older['spent'].items():
        if newer['spent'][stage] < value:
            return False
    if newer['stage'] == older['stage']:
        return newer['stage_iteration'] >= older['stage_iteration']
    try:
        index = STAGE_ORDER.index(older['stage'])
    except ValueError:
        return False
    if index + 1 >= len(STAGE_ORDER) or newer['stage'] != STAGE_ORDER[index + 1]:
        return False
    return newer['stage_iteration'] == 0


def reconcile_state(*, state_path, ledger, directory=None, load=None):
    """Atomically synchronise a run state that is behind the durable ledger.

    SDS 7.2: when several records exist for one update the ledger's verified
    complete-boundary order decides (never file mtime); if the run state is
    behind and the newest record is *uniquely* determined and compatible, the
    state and its checkpoint reference are atomically rewritten before the
    resume, otherwise the resume is blocked.
    """
    directory = Path(directory) if directory is not None else Path(state_path).parent
    selected = select_checkpoint(ledger, directory=directory, load=load)
    checkpoint = load(selected['path'], selected['sha256']) if load is not None else None
    latest = ledger.completed_ppo_updates
    try:
        existing = read_state(state_path, mode=ledger.mode)
    except (ValueError, LegacySchemaError) as error:
        raise RecoveryBlocked(f'run state file is not a valid v0.5-r2 state: {error}') from error
    if existing is not None:
        reference = existing['checkpoint']
        anchor = existing['ledger_anchor']
        if not isinstance(anchor, dict) or set(anchor) != {'seq', 'digest'}:
            raise RecoveryBlocked('run state ledger anchor schema mismatch')
        if anchor['seq'] > ledger.anchor()['seq']:
            raise RecoveryBlocked('run state references a durable ledger tail that does not exist')
        if anchor['seq'] and ledger.events[anchor['seq'] - 1]['digest'] != anchor['digest']:
            raise RecoveryBlocked('run state ledger anchor is not a prefix of this ledger')
        if existing['learning']['update'] > latest:
            raise RecoveryBlocked('run state is ahead of the durable ledger')
        if reference['update'] > selected['update']:
            raise RecoveryBlocked('refusing to move the run state back to an older checkpoint')
        if reference['sha256'] != selected['sha256']:
            if reference['update'] > latest:
                raise RecoveryBlocked('run state checkpoint is newer than the durable ledger')
        if not _reachable(existing['curriculum'], ledger.curriculum):
            raise RecoveryBlocked('run state curriculum is not a monotone prefix of the ledger')
    if checkpoint is not None:
        if checkpoint['learning']['update'] != selected['update']:
            raise RecoveryBlocked('selected checkpoint learning boundary mismatch')
        if not _reachable(checkpoint['curriculum'], ledger.curriculum):
            raise RecoveryBlocked('checkpoint curriculum is not a monotone prefix of the ledger')
    new_state = dict(schema=SCHEMA_VERSION, run_id=ledger.header['run_id'],
                     learning=deepcopy(ledger.expected_learning()),
                     curriculum=deepcopy(ledger.curriculum),
                     checkpoint=dict(path=str(selected['path']), sha256=selected['sha256'],
                                     update=selected['update'],
                                     ledger_anchor=deepcopy(selected['anchor'])),
                     ledger_anchor=ledger.anchor(),
                     evaluation=ledger.checkpoint_evaluation())
    if existing == new_state:
        status = 'up_to_date'
    else:
        atomic_json(state_path, new_state)
        status = 'synchronized' if existing is not None else 'created'
    audit_budget(new_state['learning'], new_state['curriculum'], mode=ledger.mode)
    plan = resume_plan(ledger)
    plan.update(status=status, checkpoint=dict(new_state['checkpoint']),
                learning=new_state['learning'], curriculum=new_state['curriculum'],
                evaluation=new_state['evaluation'], state_path=str(Path(state_path)),
                ledger_anchor=new_state['ledger_anchor'])
    return plan


# ------------------------------------------------------------- restore order
class RecoverySequence:
    """The only legal SDS 7.7 restore order, enforced step by step.

    ``construct`` happens before this object: the caller builds the environment
    and model (any reset performed by construction is discarded by the forced
    new-episode reset at the end).  Then, exactly:

    load learning -> bind curriculum/slots (no implicit reset) -> restore all
    RNG (python/numpy/torch/cuda + env generator) -> one explicit new-episode
    reset -> refresh the runner observation.

    A second reset, a reset before the RNG restore, a re-seed after the restore
    or a curriculum bind that resets are all hard errors.
    """
    STEPS = ('constructed', 'learning', 'curriculum', 'rng', 'ready')

    def __init__(self, *, env, model, optimizer=None, observation_getter=None,
                 bind_stage=None):
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.observation_getter = observation_getter
        self.bind_stage = bind_stage or (lambda stage, iteration: env.set_stage(stage, iteration))
        self.state = 'constructed'
        self.learning = None
        self.curriculum = None
        self.obs = None
        self.reset_count = 0
        self.construction_reset_count = int(getattr(env, 'reset_count', 0))
        self.rng_captured = None

    def _require(self, expected):
        if self.state != expected:
            raise ValueError(f'recovery order violation: expected step {expected!r}, '
                             f'current {self.state!r}')

    def load_learning(self, checkpoint):
        self._require('constructed')
        self.model.load_state_dict(checkpoint['model'], strict=True)
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        normalizer = checkpoint.get('normalizer')
        if normalizer:
            target = getattr(self.model, 'normalizer', None)
            if target is None:
                actor = getattr(self.model, 'actor', None)
                target = getattr(actor, 'normalizer', None) if actor is not None else None
            if target is None or not hasattr(target, 'load_state_dict'):
                raise ValueError('model exposes no normalizer holder for the frozen normalizer')
            target.load_state_dict(normalizer)
        self.learning = deepcopy(checkpoint['learning'])
        validate_learning(self.learning)
        self.state = 'learning'
        return self

    def bind_curriculum(self, checkpoint):
        self._require('learning')
        curriculum = deepcopy(checkpoint['curriculum'])
        validate_curriculum(curriculum, learning=checkpoint['learning'],
                            mode=checkpoint['source']['mode'])
        before = int(getattr(self.env, 'reset_count', 0))
        self.bind_stage(curriculum['stage'], curriculum['stage_iteration'])
        after = int(getattr(self.env, 'reset_count', 0))
        if after != before:
            raise ValueError('curriculum/slot binding must not reset the environment')
        self.curriculum = curriculum
        self.state = 'curriculum'
        return self

    def restore_rng(self, checkpoint):
        self._require('curriculum')
        restore_rng(checkpoint['rng'], self.env)
        self.rng_captured = capture_rng(self.env)
        self.state = 'rng'
        return self

    def new_episode_reset(self):
        self._require('rng')
        before = int(getattr(self.env, 'reset_count', 0))
        result = self.env.reset()
        observation = result[0] if isinstance(result, tuple) else result
        after = int(getattr(self.env, 'reset_count', 0))
        if after != before + 1:
            raise ValueError('exactly one explicit new-episode reset is required')
        self.reset_count = after - self.construction_reset_count
        self.obs = (self.observation_getter() if self.observation_getter is not None
                    else observation)
        self.state = 'ready'
        return self.obs


def claim_worker_directory(root, *, prefix='attempt'):
    """Create the next monotonic attempt directory; never overwrite an existing one."""
    root = Path(root)
    if not isinstance(prefix, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', prefix):
        raise ValueError('attempt prefix must be a simple directory name')
    root.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf'^{re.escape(prefix)}-(\d{{4,}})$')
    numbers = [int(match.group(1)) for match in
               (pattern.match(path.name) for path in root.iterdir()) if match]
    attempt = max(numbers, default=0) + 1
    while True:
        candidate = root / f'{prefix}-{attempt:04d}'
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            attempt += 1
            continue
        return candidate
