"""v0.6 orchestration: preflight, admission, single-worker execution, resources.

SDS 11.2 contract:

* ``--execute`` absent => **preflight only**: nothing is created under
  ``--output-dir``, no CUDA context is opened and no W&B session starts;
* an existing ``--output-dir`` is refused, never overwritten;
* exactly one GPU worker may run, and the four resource limits (whole card
  <=7168 MiB, training process <=5120 MiB, host available >=4096 MiB, disk
  >=10 GiB) are enforced before and during the run;
* every CLI number is an assertion against the frozen recipe, never an override;
* ``--resume-run`` is the only resume interface and it re-validates the original
  contract instead of re-initializing from ETH.

All numeric limits are read from the frozen spec module when it exposes them;
otherwise the SDS fallback is used and recorded in the preflight evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

from . import ledger as ledger_module
from . import model as model_module
from . import ppo as ppo_module
from . import recovery as recovery_module
from . import speed

#: Project root: ``src/tasks/velocity/v07/orchestration.py`` -> parents[4].
ROOT = Path(__file__).resolve().parents[4]

PHASE_BUDGET_ORDER = ('F0', 'H', 'D', 'L', 'V', 'R')
#: SDS 8.5 resource ceilings (fallback when spec does not expose them yet).
_RESOURCE_FALLBACK = dict(whole_device_mib=7168, process_mib=5120,
                          host_available_mib=4096, disk_free_gib=10)
#: SDS 5.4 command contract (fallback when spec.CURRICULUM omits it).
_COMMAND_FALLBACK = dict(initial_vx=(0.30, 0.80), final_vx=(0.20, 0.60), vy=0.0, wz=0.0)
DEFAULT_PREFLIGHT_DIR = 'logs/a2_ideal_stairs_v07/implementation/preflight'
#: SDS 9.3 probe evidence root; never the training run directory.
PROBE_DIR = 'logs/a2_ideal_stairs_v07/implementation/probe'
PROBE_TILE = 'L40'
#: Probe checkpoint container tag (read from the frozen spec lazily).
PROBE_CONTAINER_FALLBACK = 'v07-probe-state-v1'
PROBE_ACCEPTANCE_SCOPE = 'engineering_check_only'
MIB = 1024 ** 2
GIB = 1024 ** 3


class PreflightError(ValueError):
    """A CLI value or an environment fact contradicts the frozen v0.6 recipe."""


class ProbeError(RuntimeError):
    """The resource/recovery probe plan, execution or evidence is invalid."""


class AdmissionError(RuntimeError):
    """Execution is refused (existing directory, busy worker, resources)."""


def _resource_limits():
    raw = getattr(spec_module(), 'RESOURCE_LIMITS', None) \
        or getattr(spec_module(), 'W7_RESOURCE_LIMITS', None)
    limits, source = dict(_RESOURCE_FALLBACK), 'fallback'
    if isinstance(raw, dict):
        aliases = dict(whole_device_mib=('whole_device_mib', 'gpu_mib', 'card_mib',
                                         'device_mib'),
                       process_mib=('process_mib', 'process_memory_mib'),
                       host_available_mib=('host_available_mib', 'host_free_mib',
                                           'host_available'),
                       disk_free_gib=('disk_free_gib', 'disk_gib', 'disk_free'))
        for name, keys in aliases.items():
            for key in keys:
                if key in raw:
                    limits[name] = raw[key]
                    source = 'spec.RESOURCE_LIMITS'
                    break
    return limits, source


def _command_ranges():
    spec = spec_module()
    raw = getattr(spec, 'CURRICULUM', {})
    settings, source = dict(_COMMAND_FALLBACK), 'fallback'
    if isinstance(raw, dict):
        command_block = raw.get('command') if isinstance(raw.get('command'), dict) else {}
        for name, keys in (('initial_vx', ('initial_vx', 'command_initial', 'vx_initial')),
                           ('final_vx', ('final_vx', 'command_final', 'vx_final'))):
            for key in keys:
                value = raw.get(key, command_block.get(key))
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    settings[name] = (float(value[0]), float(value[1]))
                    source = 'spec.CURRICULUM'
                    break
        for name in ('vy', 'wz'):
            if name in raw or name in command_block:
                settings[name] = float(raw.get(name, command_block.get(name)))
                source = 'spec.CURRICULUM'
    return settings, source


def spec_module():
    from . import spec
    return spec


def probe_container():
    return getattr(spec_module(), 'PROBE_CONTAINER', PROBE_CONTAINER_FALLBACK)


def resource_limits():
    return _resource_limits()


def command_ranges():
    return _command_ranges()


# ------------------------------------------------------------------- probe
def probe_plan(*, num_envs=None, eval_num_envs=None, seed=42):
    """The SDS 9.3 resource/recovery probe plan and its exact cost identity.

    512 training environments + 8 evaluation environments, the final L40 terrain,
    5 warmup + 50 measured + 1 recovery update = 56 learning updates.  The cost is
    billed separately and never touches the 30,000-update training budget.
    """
    spec = spec_module()
    envs = int(spec.PROBE_ENVS if num_envs is None else num_envs)
    evaluation_envs = int(spec.PROBE_EVAL_ENVS if eval_num_envs is None
                          else eval_num_envs)
    warmup = int(spec.PROBE_WARMUP_UPDATES)
    measured = int(spec.PROBE_MEASURE_UPDATES)
    recovery = int(spec.PROBE_RECOVERY_UPDATES)
    updates = warmup + measured + recovery
    if envs != int(spec.PROBE_ENVS):
        raise ProbeError(f'the probe trains with {spec.PROBE_ENVS} environments, not {envs}')
    if evaluation_envs != int(spec.PROBE_EVAL_ENVS):
        raise ProbeError(f'the probe evaluates with {spec.PROBE_EVAL_ENVS} environments')
    if updates != int(spec.PROBE_UPDATES):
        raise ProbeError('the probe update stages do not sum to the frozen 56 updates')
    rollout = int(spec.PPO_ROLLOUT)
    control_samples = envs * rollout * updates
    physics = ppo_module.physics_substeps(control_samples)
    adam = updates * int(spec.ADAM_UPDATES_PER_ITERATION)
    if control_samples != int(spec.PROBE_CONTROL_SAMPLES):
        raise ProbeError('probe control-sample cost differs from the frozen 1,376,256')
    train_total = int(spec.total_budget())
    return dict(
        seed=int(seed), num_envs=envs, eval_num_envs=evaluation_envs,
        tile=PROBE_TILE, tile_id=int(spec.FINAL_TILE_ID), phase='R',
        stages=dict(warmup=warmup, measured=measured, recovery=recovery),
        updates=updates, rollout=rollout,
        epochs=int(spec.PPO_EPOCHS), minibatches=int(spec.PPO_MINIBATCHES),
        cost=dict(control_samples=control_samples, physics_substeps=physics,
                  adam_updates=adam, learning_updates=updates),
        measurements=dict(warmup=warmup, measured=measured, recovery=recovery),
        resource_limits=dict(_resource_limits()[0]),
        counts_against_training=False,
        training_budget_untouched=train_total,
        training_budget_consumed=0,
        container=probe_container(), resumable=False,
        acceptance_scope=PROBE_ACCEPTANCE_SCOPE,
        notes='probe evidence may not initialize or resume a formal v0.6 run')


def probe_ledger(plan):
    """The probe's own billed-progress ledger, isolated from the training budget."""
    plan = plan or probe_plan()
    return dict(
        learning_updates=int(plan['cost']['learning_updates']),
        control_samples=int(plan['cost']['control_samples']),
        physics_substeps=int(plan['cost']['physics_substeps']),
        adam_updates=int(plan['cost']['adam_updates']),
        counts_against_training=False,
        training_budget_untouched=int(plan['training_budget_untouched']),
        training_budget_consumed=0,
        container=probe_container(), resumable=False,
        warmup=int(plan['stages']['warmup']),
        measured=int(plan['stages']['measured']),
        recovery=int(plan['stages']['recovery']))


class ProbeLedger:
    """Progress bookkeeping for the 56-update probe (fake-friendly, CPU-only).

    A recovery resumes from update 55 with exactly one remaining update; a replayed
    or duplicated update is refused, so "one recovery update" is enforced rather
    than asserted after the fact.
    """

    def __init__(self, plan=None):
        self.plan = dict(plan or probe_plan())
        self.ledger = probe_ledger(self.plan)
        self.recorded = []
        self.resumed_from = None
        self.measured_samples = []

    @property
    def expected_updates(self):
        return int(self.plan['cost']['learning_updates'])

    def record_update(self, index, *, samples=0):
        index = int(index)
        if index in self.recorded:
            raise ProbeError(f'probe update {index} would be billed twice')
        if index != len(self.recorded) + 1 and self.resumed_from is None:
            raise ProbeError(f'probe update {index} is out of order')
        self.recorded.append(index)
        if samples:
            self.measured_samples.append(dict(samples))
        return index

    def resume(self, index):
        """Continue at ``index`` after a checkpoint; nothing before it is re-run."""
        index = int(index)
        if self.resumed_from is not None:
            raise ProbeError('the probe already resumed once')
        if index != len(self.recorded) + 1:
            raise ProbeError(
                f'resume must continue at {len(self.recorded) + 1}, not {index}')
        self.resumed_from = index
        return dict(resumed_from_update=len(self.recorded),
                    next_update=index, updates_after=1,
                    billed_updates_not_replayed=True)

    def receipt(self):
        return dict(updates=len(self.recorded), expected_updates=self.expected_updates,
                    resumed_from_update=(None if self.resumed_from is None
                                         else self.resumed_from - 1),
                    updates_after_recovery=(None if self.resumed_from is None
                                            else len(self.recorded) - self.resumed_from + 1),
                    ledger=dict(self.ledger))

    def __call__(self, index, *, samples=0):
        return self.record_update(index, samples=samples)


def summarize_resource_samples(samples, limits=None):
    """min/max/median of the four SDS resource metrics over the measured window."""
    if not samples:
        raise ProbeError('the probe measured no resource samples')
    limits = dict(limits or _resource_limits()[0])
    keys = (('whole_device_mib', 7168, 'max'), ('process_mib', 5120, 'max'),
            ('host_available_mib', 4096, 'min'), ('disk_free_gib', 10.0, 'min'))
    summary, blocking = {}, []
    for name, _default, direction in keys:
        values = [float(sample[name]) for sample in samples
                  if isinstance(sample, dict) and sample.get(name) is not None]
        if not values:
            summary[name] = dict(min=None, max=None, median=None, limit=limits[name],
                                 direction=direction, ok=False,
                                 **(dict(method=None) if name == 'process_mib' else {}))
            blocking.append(name)
            continue
        values.sort()
        middle = len(values) // 2
        median = (values[middle] if len(values) % 2
                  else 0.5 * (values[middle - 1] + values[middle]))
        limit = float(limits[name])
        ok = (max(values) <= limit) if direction == 'max' else (min(values) >= limit)
        entry = dict(min=values[0], max=values[-1], median=median,
                     limit=limit, direction=direction, ok=bool(ok),
                     samples=len(values))
        if name == 'process_mib':
            methods = [sample.get('process_mib_method') for sample in samples
                       if isinstance(sample, dict)
                       and sample.get('process_mib_method')]
            entry['method'] = (methods[0] if methods and len(set(methods)) == 1
                               else ('mixed' if methods else None))
        summary[name] = entry
        if not ok:
            blocking.append(name)
    return dict(metrics=summary, blocking=blocking, ok=not blocking,
                samples=len(samples), limits=limits,
                acceptance_scope=PROBE_ACCEPTANCE_SCOPE)


def validate_probe_recovery(receipt, plan=None):
    """The probe's resume receipt must prove a single non-replayed recovery update."""
    plan = plan or probe_plan()
    if not isinstance(receipt, dict):
        raise ProbeError('the probe returned no recovery receipt')
    required = ('resume_mode', 'resumed_from_update', 'updates_after',
                'billed_updates_not_replayed', 'initialized_from_eth_again',
                'container', 'checkpoint_sha256', 'curriculum_state', 'counts')
    missing = [key for key in required if key not in receipt]
    if missing:
        raise ProbeError(f'probe recovery receipt is missing {missing}')
    if receipt['resume_mode'] != 'new_episode':
        raise ProbeError('the probe recovery must start a new episode')
    if int(receipt['resumed_from_update']) != int(plan['stages']['warmup'] +
                                                 plan['stages']['measured']):
        raise ProbeError('the probe checkpoint must be saved after update 55')
    if int(receipt['updates_after']) != int(plan['stages']['recovery']):
        raise ProbeError('the probe recovery must bill exactly one update')
    if receipt['billed_updates_not_replayed'] is not True:
        raise ProbeError('the probe recovery replayed an already billed update')
    if receipt['initialized_from_eth_again'] is not False:
        raise ProbeError('the probe recovery re-initialized from ETH')
    if receipt['container'] != probe_container():
        raise ProbeError('the probe checkpoint lacks the probe container tag')
    counts = receipt['counts']
    if int(counts.get('learning_updates', -1)) != int(plan['cost']['learning_updates']):
        raise ProbeError('the probe recovery counts do not reach the full 56 updates')
    if int(counts.get('control_samples', -1)) != int(plan['cost']['control_samples']):
        raise ProbeError('the probe recovery control-sample count is wrong')
    if counts.get('counts_against_training') is not False:
        raise ProbeError('probe updates must not count against the training budget')
    return True


# ------------------------------------------------------------------- parser
def build_parser():
    """The exact SDS 11.2 CLI surface (no flag is silently dropped)."""
    parser = argparse.ArgumentParser(
        description='A2 ideal-heightmap stairs v0.6 single continuous training run',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', default='train', choices=('train', 'probe'))
    parser.add_argument('--execute', action='store_true',
                        help='start the GPU worker; without it only a preflight runs')
    parser.add_argument('--recipe', default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--num-envs', type=int, default=None)
    parser.add_argument('--eval-num-envs', type=int, default=None)
    parser.add_argument('--rollout', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--minibatches', type=int, default=None)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--entropy-coef', type=float, default=None)
    parser.add_argument('--initial-std', type=float, default=None)
    parser.add_argument('--std-min', type=float, default=None)
    parser.add_argument('--std-max', type=float, default=None)
    parser.add_argument('--initial-vx-min', type=float, default=None)
    parser.add_argument('--initial-vx-max', type=float, default=None)
    parser.add_argument('--final-vx-min', type=float, default=None)
    parser.add_argument('--final-vx-max', type=float, default=None)
    parser.add_argument('--vy', type=float, default=None)
    parser.add_argument('--wz', type=float, default=None)
    parser.add_argument('--max-updates', type=int, default=None)
    parser.add_argument('--phase-budgets', default=None,
                        help='F0=1000,H=12000,D=3000,L=6000,V=4000,R=4000')
    parser.add_argument('--save-interval', type=int, default=None)
    parser.add_argument('--diagnostic-interval', type=int, default=None)
    parser.add_argument('--gate-interval', type=int, default=None)
    parser.add_argument('--init-checkpoint', default=None)
    parser.add_argument('--init-sha256', default=None)
    parser.add_argument('--acceptance-file', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--allow-q005-bridge', action='store_true',
                        help='restricted Q005 bridge for the documented trial02 '
                             'boundary checkpoint only')
    parser.add_argument('--allow-q006-bridge', action='store_true',
                        help='Q006 extension of the same restricted bridge: the lost '
                             '101-150 interval is replayed and billed (30100 ceiling) '
                             'into a new _q006 segment')
    parser.add_argument('--allow-q007-bridge', action='store_true',
                        help='Q007 restricted bridge: resume the designated '
                             'exception_000400.pt, migrate only its unfinished flat '
                             'retention task and replay it once into a _q007 segment')

    parser.add_argument('--resume-run', default=None,
                        help='the only resume interface: a validated schema-8 checkpoint')
    return parser


def parse_phase_budgets(text):
    if text is None:
        return None
    values = {}
    for chunk in str(text).split(','):
        if not chunk.strip():
            continue
        if '=' not in chunk:
            raise PreflightError(f'--phase-budgets entry {chunk!r} is not NAME=COUNT')
        name, _, count = chunk.partition('=')
        name = name.strip().upper()
        if name in values:
            raise PreflightError(f'--phase-budgets repeats {name}')
        try:
            values[name] = int(count)
        except ValueError:
            raise PreflightError(f'--phase-budgets entry {chunk!r} is not an integer') from None
    return values


def expected_phase_budgets():
    return {phase: speed.phase_budget(phase) for phase in PHASE_BUDGET_ORDER}


def _check(condition, message):
    if not condition:
        raise PreflightError(message)


def _same(left, right):
    return left is None or abs(float(left) - float(right)) <= 1e-12


RECIPE_FLAGS = ('mode', 'execute', 'recipe', 'seed', 'device', 'num_envs',
                'eval_num_envs', 'rollout', 'epochs', 'minibatches', 'learning_rate',
                'gamma', 'entropy_coef', 'initial_std', 'std_min', 'std_max',
                'initial_vx_min', 'initial_vx_max', 'final_vx_min', 'final_vx_max',
                'vy', 'wz', 'max_updates', 'phase_budgets', 'save_interval',
                'diagnostic_interval', 'gate_interval', 'init_checkpoint',
                'init_sha256', 'acceptance_file', 'output_dir')
#: Run-local paths are not recipe numbers: the SDS fixes the *acceptance* path but
#: the operator may pick another output directory (an existing one is refused).
RUN_LOCAL_FLAGS = ('output_dir', 'acceptance_file')
#: Flags that may not accompany ``--resume-run`` (it must never re-initialize).
RESUME_EXCLUDED = ('init_checkpoint', 'init_sha256', 'phase_budgets', 'max_updates',
                   'seed', 'num_envs', 'device', 'recipe')


def _provided(args):
    """The recipe flags the operator actually supplied (``None`` = not supplied)."""
    provided = {}
    for name in RECIPE_FLAGS:
        value = getattr(args, name)
        if name == 'mode':
            # probe borrows the same frozen recipe identity/config as training.
            provided[name] = 'train'
        elif name == 'execute':
            provided[name] = bool(value)
        elif value is not None and name not in RUN_LOCAL_FLAGS:
            provided[name] = value
    return provided


def _rooted(path, root):
    """Resolve a run-local path against the project root, not the process CWD."""
    path = Path(path)
    return path if path.is_absolute() else (Path(root) / path)


def validate_arguments(args, *, require_acceptance=True, root=None):
    """Assert every SDS 11.2 flag against ``spec.recipe_config`` and normalize it.

    ``require_acceptance`` exists so CPU tests can exercise admission without the
    reviewer-written acceptance report; every production call leaves it on.
    """
    spec = spec_module()
    if args.mode not in ('train', 'probe'):
        raise PreflightError('v0.6 supports --mode train and --mode probe only')
    if args.mode == 'probe' and args.resume_run is not None:
        raise PreflightError(
            '--resume-run is the formal training resume interface; the probe performs '
            'its own recovery verification internally')
    provided = _provided(args)
    q007_bridge = bool(getattr(args, 'allow_q007_bridge', False))
    bridge_flag = bool(getattr(args, 'allow_q005_bridge', False)
                       or getattr(args, 'allow_q006_bridge', False)
                       or q007_bridge)
    if bridge_flag and args.resume_run is None:
        raise PreflightError('--allow-q005-bridge/--allow-q006-bridge/'
                             '--allow-q007-bridge requires --resume-run')
    if args.resume_run is not None:
        resume_path = Path(args.resume_run)
        _check(resume_path.is_file(), f'--resume-run checkpoint does not exist: {resume_path}')
        for name in RESUME_EXCLUDED:
            _check(name not in provided or name in ('execute',),
                   f'--resume-run must not receive --{name.replace("_", "-")} '
                   f'(the contract comes from the checkpoint, never from ETH)')
        # Resume still asserts the *training* contract values that define the run.
        provided = {key: value for key, value in provided.items()
                    if key in ('mode', 'execute', 'recipe', 'gamma', 'entropy_coef',
                               'initial_std', 'std_min', 'std_max', 'learning_rate',
                               'rollout', 'epochs', 'minibatches', 'eval_num_envs',
                               'save_interval', 'diagnostic_interval', 'gate_interval')}
    try:
        resolved = spec.recipe_config(**provided)
    except ValueError as error:
        raise PreflightError(str(error)) from None
    init_checkpoint = str(resolved['init_checkpoint'])
    init_sha = str(resolved['init_sha256'])
    if args.resume_run is None:
        _check(Path(init_checkpoint).is_file(),
               f'--init-checkpoint does not exist: {init_checkpoint}')
        _check(model_module.sha256_file(init_checkpoint) == model_module.INIT_SHA256,
               '--init-checkpoint does not match the approved ETH SHA256')
        _check(init_sha == model_module.INIT_SHA256,
               '--init-sha256 is not the approved ETH SHA256')
    root = Path(root or ROOT)
    if args.mode == 'probe':
        probe_root = root / PROBE_DIR
        output_dir = (_rooted(args.output_dir, root) if args.output_dir
                      else probe_root / f'probe_seed{int(resolved["seed"])}')
        try:
            output_dir.resolve().relative_to(probe_root.resolve())
        except ValueError:
            raise PreflightError(
                f'a probe run must write under {probe_root}, not {output_dir}') from None
        _check(output_dir.resolve() != (root / 'logs/a2_ideal_stairs_v07/train_seed42_trial01'
                                        ).resolve(),
               'the probe must never reuse the formal training run directory')
    else:
        if (args.resume_run is not None and args.output_dir is None
                and args.mode != 'probe'):
            _resume_state = recovery_module.preload_checkpoint(
                args.resume_run, mode='resume', allow_bridge=bridge_flag)
            _update = int(_resume_state['learning']['update'])
            # Q006 section 1: the lossy rollback replay writes ONE new segment
            # (``_q006``); the existing Q005 segment and its bytes are untouched.
            if q007_bridge:
                suffix = ledger_module.Q007_SEGMENT_SUFFIX
            elif bridge_flag:
                suffix = ledger_module.Q006_SEGMENT_SUFFIX
            else:
                suffix = ''
            output_dir = (Path(root) / 'logs/a2_ideal_stairs_v07'
                          / f'train_seed42_trial02_resume_{_update:06d}{suffix}')
        else:
            output_dir = _rooted(args.output_dir or resolved['output_dir'], root)
    acceptance_name = args.acceptance_file or resolved.get('acceptance_file')
    acceptance = _rooted(acceptance_name, root) if acceptance_name else None
    if (acceptance is not None and require_acceptance and resolved['execute']
            and args.resume_run is None and args.mode != 'probe'):
        _check(acceptance.is_file(),
               f'--acceptance-file does not exist yet: {acceptance}')
    limits, limits_source = _resource_limits()
    commands, command_source = _command_ranges()
    # Q001-7: the inherited stall-collapse streak is a required frozen constant;
    # a missing/invalid value stops the preflight instead of defaulting.
    collapse = speed.require_diagnostic_collapse_streak()
    contract = dict(
        mode=('resume' if args.resume_run is not None
              else ('probe' if args.mode == 'probe' else 'train')),
        execute=bool(resolved['execute']), recipe=resolved['recipe'],
        seed=int(resolved['seed']), device=str(resolved['device']),
        num_envs=int(resolved['num_envs']),
        eval_num_envs=int(resolved['eval_num_envs']),
        rollout=int(resolved['rollout']), epochs=int(resolved['epochs']),
        minibatches=int(resolved['minibatches']),
        learning_rate=float(resolved['learning_rate']), gamma=float(resolved['gamma']),
        entropy_coef=float(resolved['entropy_coef']),
        initial_std=float(resolved['initial_std']), std_min=float(resolved['std_min']),
        std_max=float(resolved['std_max']),
        initial_vx=(float(resolved['initial_vx_min']), float(resolved['initial_vx_max'])),
        final_vx=(float(resolved['final_vx_min']), float(resolved['final_vx_max'])),
        vy=float(resolved['vy']), wz=float(resolved['wz']),
        max_updates=int(resolved['max_updates']),
        phase_budgets=dict(resolved['phase_budgets']),
        save_interval=int(resolved['save_interval']),
        diagnostic_interval=int(resolved['diagnostic_interval']),
        gate_interval=int(resolved['gate_interval']),
        init_checkpoint=init_checkpoint, init_sha256=init_sha,
        acceptance_file=str(acceptance) if acceptance else None,
        output_dir=str(output_dir),
        resume_run=str(args.resume_run) if args.resume_run else None,
        allow_q005_bridge=bool(args.allow_q005_bridge),
        allow_q006_bridge=bool(getattr(args, 'allow_q006_bridge', False)),
        allow_q007_bridge=q007_bridge,
        bridge_flag=bridge_flag,
        segment=((ledger_module.q007_segment_metadata() if q007_bridge
                  else ledger_module.segment_metadata())
                 if (bridge_flag and args.resume_run is not None) else None),
        provided=provided, resolved_recipe=dict(resolved),
        command_source=command_source,
        resource_limits=limits, resource_limits_source=limits_source,
        diagnostic_collapse_streak=collapse,
        identity=spec.identity_block(),
    )
    if args.mode == 'probe':
        contract['probe'] = probe_plan(seed=int(resolved['seed']),
                                       num_envs=int(resolved['num_envs']),
                                       eval_num_envs=int(resolved['eval_num_envs']))
        contract['probe']['device'] = str(resolved['device'])
        contract['probe_root'] = str(root / PROBE_DIR)
        contract['probe_container'] = probe_container()
        contract['probe_resumable'] = False
        contract['counts_against_training'] = False
        contract['training_budget_untouched'] = int(speed.total_budget())
    contract['resume'] = None
    if args.resume_run is not None:
        allow_bridge = bridge_flag
        try:
            state = recovery_module.preload_checkpoint(
                args.resume_run, mode='resume', allow_bridge=allow_bridge)
        except recovery_module.RecoveryBlocked as error:
            raise PreflightError(str(error)) from None
        bridge_manifest = None
        q006_manifest = None
        if getattr(args, 'allow_q005_bridge', False):
            manifest = (Path(root) / 'logs/a2_ideal_stairs_v07/implementation/'
                        'coordination/Q005_recovery_bridge.md')
            if not manifest.is_file():
                raise PreflightError('Q005 recovery manifest is missing')
            text = manifest.read_text(encoding='utf-8')
            for label, value in (
                    ('checkpoint', recovery_module.BRIDGE_CHECKPOINT_SHA256),
                    ('old closure', recovery_module.BRIDGE_OLD_SOURCE_CLOSURE_SHA256),
                    ('Q005', recovery_module.Q005_DECISION_SHA256)):
                if value not in text:
                    raise PreflightError(f'Q005 manifest does not record the {label} SHA')
            bridge_manifest = dict(path=str(manifest),
                                   sha256=recovery_module.sha256(manifest))
        if getattr(args, 'allow_q006_bridge', False):
            manifest = Path(root) / ledger_module.Q006_BRIDGE_MANIFEST_PATH
            if not manifest.is_file():
                raise PreflightError('Q006 recovery manifest is missing')
            text = manifest.read_text(encoding='utf-8')
            for label, value in (
                    ('checkpoint', recovery_module.BRIDGE_CHECKPOINT_SHA256),
                    ('old closure', recovery_module.BRIDGE_OLD_SOURCE_CLOSURE_SHA256),
                    ('Q005', recovery_module.Q005_DECISION_SHA256),
                    ('Q006', ledger_module.Q006_DECISION_SHA256),
                    ('Q006 errata', ledger_module.Q006_ERRATA_SHA256)):
                if value not in text:
                    raise PreflightError(f'Q006 manifest does not record the {label} SHA')
            q006_manifest = dict(path=str(manifest),
                                 sha256=recovery_module.sha256(manifest))
        q007_manifest = None
        if q007_bridge:
            manifest = Path(root) / ledger_module.Q007_BRIDGE_MANIFEST_PATH
            if not manifest.is_file():
                raise PreflightError('Q007 recovery manifest is missing')
            text = manifest.read_text(encoding='utf-8')
            for label, value in (
                    ('source checkpoint',
                     ledger_module.Q007_SOURCE_CHECKPOINT_SHA256),
                    ('forbidden boundary',
                     ledger_module.Q007_FORBIDDEN_CHECKPOINT_SHA256),
                    ('old closure', ledger_module.Q007_OLD_SOURCE_CLOSURE_SHA256),
                    ('Q006', ledger_module.Q006_DECISION_SHA256),
                    ('Q007', ledger_module.Q007_DECISION_SHA256),
                    ('Q007 errata', ledger_module.Q007_ERRATA_SHA256)):
                if value not in text:
                    raise PreflightError(
                        f'Q007 manifest does not record the {label} SHA')
            q007_manifest = dict(path=str(manifest),
                                 sha256=recovery_module.sha256(manifest))
        contract['allow_bridge'] = allow_bridge
        contract['bridge_manifest'] = bridge_manifest
        contract['q006_bridge_manifest'] = q006_manifest
        contract['q007_bridge_manifest'] = q007_manifest
        contract['resume'] = recovery_module.resume_contract(
            state, recipe=resolved['recipe'], allow_bridge=allow_bridge)
        if allow_bridge and not (contract['resume'].get('bridge') or {}).get('restricted'):
            raise PreflightError('the restricted bridge was not confirmed for this '
                                 'checkpoint')
        if contract.get('segment') is not None:
            if (int(contract['segment']['resume_from_update'])
                    != int(contract['resume']['update'])):
                raise PreflightError(
                    'the segment metadata does not match the checkpoint update')
            if q007_bridge:
                if contract['resume'].get('replay_billed_updates') is not False:
                    raise PreflightError(
                        'the Q007 bridge must declare replay_billed_updates=false')
                if int(contract['segment']['lost_updates']) != 0:
                    raise PreflightError('the Q007 bridge loses no learning update')
                migration = contract['resume'].get('pending_task_migration')
                if not isinstance(migration, dict) or \
                        migration.get('q007_decision_sha256') != \
                        ledger_module.Q007_DECISION_SHA256:
                    raise PreflightError(
                        'the Q007 bridge did not produce the pending-task migration')
            elif not contract['resume'].get('replay_billed_updates'):
                raise PreflightError(
                    'the Q006 bridge must declare replay_billed_updates=true')
    return contract


# ----------------------------------------------------------------- resources
def _process_tree(pid):
    """The live process and its descendants (``/proc/<pid>/task/<pid>/children``)."""
    root = int(pid)
    tree, pending = {root}, [root]
    while pending:
        current = pending.pop()
        try:
            with open(f'/proc/{current}/task/{current}/children', encoding='utf-8') as handle:
                children = [int(value) for value in handle.read().split()]
        except (OSError, ValueError):
            continue
        for child in children:
            if child not in tree:
                tree.add(child)
                pending.append(child)
    return tree


def _torch_reserved_mib():
    """This process's CUDA reserved memory, without initializing a CUDA context."""
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return None
        values = [float(torch.cuda.max_memory_reserved(index)) / MIB
                  for index in range(torch.cuda.device_count())]
    except (RuntimeError, AssertionError, ValueError):
        return None
    return max(values) if values else None


def resource_snapshot(directory=None, *, device=0, query=None, processes=None,
                      pid=None, torch_reserved=None, stage='measured'):
    """Whole-card/process/host-available/disk snapshot in the SDS units.

    The training-process memory comes from ``nvidia-smi --query-compute-apps``
    matched to this process tree (``process_mib_method='compute_apps'``); when the
    compute-apps table cannot identify the process, the CUDA reserved-memory
    counter is used (``process_mib_method='torch_reserved'``).  Neither available
    => ``process_mib=None`` with the method recorded, so the resource guard fails
    honestly instead of silently passing.
    """
    pid = int(os.getpid() if pid is None else pid)
    snapshot = dict(timestamp=time.time(), device=int(device), pid=pid)
    snapshot.update(host_available_mib=_host_available_mib(),
                    disk_free_gib=_disk_free_gib(directory),
                    whole_device_mib=None, process_mib=None,
                    process_mib_method=None, gpu_query=[], compute_apps=[])
    try:
        if query is None and processes is None:
            tables = _nvidia_smi()
        else:
            tables = dict(gpus=list(query or []), processes=list(processes or []))
        gpus = list(tables.get('gpus') or [])
        apps = list(tables.get('processes') or [])
        # Accept legacy GPU rows that embed ``pids`` too.
        for row in gpus:
            for item in row.get('pids') or []:
                if isinstance(item, dict):
                    apps.append(dict(pid=int(item.get('pid', 0)),
                                     used_mib=int(item.get('used_mib', 0))))
                else:
                    apps.append(dict(pid=int(item), used_mib=None))
        snapshot['gpu_query'] = gpus
        snapshot['compute_apps'] = apps
        if gpus:
            snapshot['whole_device_mib'] = max(int(row.get('used_mib', 0)) for row in gpus)
        tree = _process_tree(pid)
        matched = [row for row in apps if int(row.get('pid', -1)) in tree]
        if matched:
            snapshot['process_mib'] = int(sum(int(row['used_mib']) for row in matched))
            snapshot['process_mib_method'] = 'compute_apps'
        else:
            reserved = (torch_reserved if torch_reserved is not None
                        else _torch_reserved_mib())
            if reserved is not None:
                snapshot['process_mib'] = int(round(float(reserved)))
                snapshot['process_mib_method'] = 'torch_reserved'
            elif stage == 'pre_run':
                # Before CUDA initialization the process legitimately owns no GPU
                # allocation: record the reason instead of 0 or a silent pass.
                snapshot['process_mib_method'] = 'not_initialized'
        snapshot['stage'] = str(stage)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        snapshot['gpu_error'] = repr(error)
    snapshot.setdefault('stage', str(stage))
    return snapshot


def _host_available_mib():
    try:
        with open('/proc/meminfo', encoding='utf-8') as handle:
            for line in handle:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def _disk_free_gib(directory):
    target = _rooted(directory, ROOT) if directory else ROOT
    while not target.exists() and target != target.parent:
        target = target.parent
    return shutil.disk_usage(target).free / GIB


def _nvidia_smi():
    """(gpu table, compute-apps table); process rows are ``{pid, used_mib}``."""
    query = ('--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits')
    output = subprocess.run(['nvidia-smi', *query], capture_output=True, text=True,
                            check=True).stdout
    gpus = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) < 3:
            continue
        gpus.append(dict(index=int(parts[0]), used_mib=int(float(parts[1])),
                         total_mib=int(float(parts[2]))))
    processes = []
    try:
        apps = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,used_memory',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        apps = ''
    for line in apps.splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        try:
            processes.append(dict(pid=int(parts[0]), used_mib=int(float(parts[1]))))
        except ValueError:
            continue
    return dict(gpus=gpus, processes=processes)


def check_resources(snapshot, limits=None, *, stage='measured'):
    """The four SDS 8.5 limits.

    ``stage='pre_run'`` runs before CUDA initialization: the process owns no GPU
    allocation yet, so a missing ``process_mib`` is exempted (recorded as
    ``exempted``/``not_initialized``) while whole-card, host memory and disk are
    still enforced.  The measured window (and recovery) stays fail-closed.
    """
    limits = limits or _resource_limits()[0]
    checks = dict(
        whole_device_mib=dict(limit=int(limits['whole_device_mib']),
                              value=snapshot.get('whole_device_mib')),
        process_mib=dict(limit=int(limits['process_mib']),
                         value=snapshot.get('process_mib')),
        host_available_mib=dict(limit=int(limits['host_available_mib']),
                                value=snapshot.get('host_available_mib')),
        disk_free_gib=dict(limit=float(limits['disk_free_gib']),
                           value=snapshot.get('disk_free_gib')),
    )
    exempted = []
    for name, entry in checks.items():
        if entry['value'] is None:
            if name == 'process_mib' and stage == 'pre_run':
                # Pre-run exemption: not initialized is not a violation.
                entry['ok'] = True
                entry['exempted'] = True
                entry['method'] = 'not_initialized'
                exempted.append(name)
                continue
            # An unmeasured resource is never a pass in the measured window.
            entry['ok'] = False
            continue
        entry['ok'] = (float(entry['value']) <= float(entry['limit'])
                       if name.endswith('_mib') and name.startswith(('whole', 'process'))
                       else float(entry['value']) >= float(entry['limit']))
    blocking = [name for name, entry in checks.items() if entry['ok'] is False]
    return dict(limits=dict(limits), checks=checks, blocking=blocking,
                exempted=exempted, stage=str(stage), ok=not blocking,
                snapshot=dict(snapshot))


# --------------------------------------------------------------- single worker
def worker_registry_path():
    return ROOT / DEFAULT_PREFLIGHT_DIR / 'gpu_worker.json'


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def acquire_single_worker(device, *, path=None, pid=None):
    """Claim the single GPU worker slot; refuse a second live worker."""
    path = Path(path or worker_registry_path())
    pid = int(pid or os.getpid())
    if path.exists():
        record = json.loads(path.read_text(encoding='utf-8'))
        if int(record.get('pid', 0)) != pid and pid_alive(record.get('pid', 0)):
            raise AdmissionError(
                f'another v0.6 GPU worker (pid {record.get("pid")}, device '
                f'{record.get("device")}) is already registered')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(pid=pid, device=str(device), start=time.time()),
                               indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return path


def release_single_worker(path=None, *, pid=None):
    path = Path(path or worker_registry_path())
    pid = int(pid or os.getpid())
    if path.exists():
        record = json.loads(path.read_text(encoding='utf-8'))
        if int(record.get('pid', 0)) in (pid, 0):
            path.unlink()
            return True
    return False


# ------------------------------------------------------------- orchestration
class V07Orchestrator:
    """Admission + execution owner (preflight is side-effect free)."""

    def __init__(self, args, *, root=None, preflight_dir=None, resource_probe=None,
                 worker_factory=None, probe_factory=None, gpu_registry=None,
                 clock=time.time):
        self.args = args
        self.root = _rooted(root or ROOT, ROOT)
        self.preflight_dir = _rooted(
            preflight_dir, self.root) if preflight_dir else (
                self.root / DEFAULT_PREFLIGHT_DIR)
        self.resource_probe = resource_probe or resource_snapshot
        self.worker_factory = worker_factory
        self.probe_factory = probe_factory
        self.gpu_registry = Path(gpu_registry) if gpu_registry else None
        self.clock = clock
        self.contract = None
        self.status = 'created'
        self.control = dict(action='continue', resume_authorized=False,
                            updated=float(clock()))
        self.worker_result = None
        self.resources = None

    # ------------------------------------------------------------- preflight
    def preflight(self):
        """Validate the contract and write evidence; never create the run dir."""
        self.contract = validate_arguments(self.args, root=self.root)
        output = Path(self.contract['output_dir'])
        if self.contract['execute'] and output.exists():
            raise AdmissionError(f'--output-dir already exists (refused): {output}')
        cuda_before = torch.cuda.is_initialized()
        record = dict(
            mode=self.contract['mode'], execute=bool(self.contract['execute']),
            contract=self.contract, output_dir=str(output),
            output_directory_created=False,
            output_exists=bool(output.exists()),
            cuda_initialized_before=bool(cuda_before),
            cuda_initialized_after=bool(torch.cuda.is_initialized()),
            preflight_only=not self.contract['execute'],
            probe=(dict(self.contract['probe']) if self.contract.get('probe') else None),
            sds_sha256=getattr(spec_module(), 'SDS_SHA256', None),
            recipe=spec_module().RECIPE_VERSION,
            identity=spec_module().identity_block(),
            sources=recovery_module.source_block(
                seed=self.contract['seed'], device=self.contract['device'],
                mode='train', init_path=self.contract['init_checkpoint'],
                init_sha256=self.contract['init_sha256']),
            resume_run=self.contract['resume_run'],
            timestamp=float(self.clock()),
        )
        if self.contract['resume_run']:
            # Already validated (including any restricted Q005 bridge) by
            # validate_arguments; never re-resolve it without the bridge flag.
            record['resume'] = dict(self.contract.get('resume') or {})
        self.preflight_dir.mkdir(parents=True, exist_ok=True)
        name = 'preflight_%s_seed%d.json' % (
            'probe' if self.contract['mode'] == 'probe' else 'train',
            self.contract['seed'])
        path = self.preflight_dir / name
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n',
                        encoding='utf-8')
        self.status = 'preflight_ok'
        record['path'] = str(path)
        return record

    # --------------------------------------------------------------- execute
    def execute(self):
        if self.contract is None:
            self.preflight()
        if not self.contract['execute']:
            raise AdmissionError('--execute is required to start the GPU worker')
        output = Path(self.contract['output_dir'])
        if output.exists():
            raise AdmissionError(f'--output-dir already exists (refused): {output}')
        if self.contract['mode'] == 'probe':
            return self._execute_probe(output)
        limits, _source = _resource_limits()
        # Pre-run admission happens before CUDA initialization: the process owns
        # no GPU allocation yet, so process_mib is exempted here (explicitly
        # recorded) while whole-card/host/disk are enforced.  The measured
        # post-start check below (and in the worker) stays fail-closed.
        snapshot = self.resource_probe(directory=output.parent)
        if snapshot.get('process_mib') is None \
                and snapshot.get('process_mib_method') is None:
            snapshot['process_mib_method'] = 'not_initialized'
        resources = check_resources(snapshot, limits, stage='pre_run')
        self.resources = resources
        if not resources['ok']:
            raise AdmissionError(f'insufficient resources: {resources["blocking"]}')
        registry = self.gpu_registry or (self.preflight_dir / 'gpu_worker.json')
        acquire_single_worker(self.contract['device'], path=registry)
        self.control = dict(action='continue', resume_authorized=False,
                            updated=float(self.clock()))
        output.mkdir(parents=True, exist_ok=False)
        from .operator_control import write_control
        write_control(output/'control.json',self.control)
        (output / 'contract.json').write_text(
            json.dumps(self.contract, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        if self.contract.get('segment') is not None:
            # Q006 section 1: the new segment identity is its own record; the run
            # identity stays trial02 and no old file is rewritten.
            (output / 'segment.json').write_text(
                json.dumps(self.contract['segment'], indent=2, sort_keys=True) + '\n',
                encoding='utf-8')
        (output / 'resources').mkdir(parents=True, exist_ok=True)
        (output / 'resources' / 'pre_run.json').write_text(
            json.dumps(dict(stage='pre_run', check=resources, snapshot=snapshot,
                            exempted=resources.get('exempted', []),
                            process_mib_method=resources['checks']['process_mib'].get(
                                'method')),
                       indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')
        self.status = 'running'
        try:
            if self.worker_factory is None:
                raise AdmissionError(
                    'no GPU worker factory is wired; the supervisor cannot start training')
            self.worker_result = self.worker_factory(self.contract, output)
        except BaseException as error:
            self.status = 'stopped'
            state = ('monitoring_blocked'
                     if type(error).__name__ == 'MonitoringBlocked' else None)
            # Qv07-002: keep operator bytes; failure belongs in error.json only.
            (output / 'error.json').write_text(
                json.dumps(dict(error_type=type(error).__name__, error=repr(error),
                                stop_state=state, contract=self.contract),
                           indent=2, sort_keys=True, default=str) + '\n',
                encoding='utf-8')
            raise
        finally:
            release_single_worker(registry)
        worker_state = (self.worker_result.get('state')
                        if isinstance(self.worker_result, dict) else None)
        if worker_state not in tuple(getattr(spec_module(),'STOP_STATES',())) and worker_state!='complete':
            self.control=self.read_control(output)
        elif isinstance(self.worker_result,dict):
            self.control=(self.worker_result.get('stop_evidence') or {}).get('operator_control',self.control)
        if worker_state in tuple(getattr(spec_module(), 'ALL_TRAINING_STATES', ())):
            # A measured post-start resource stop (or any terminal worker state)
            # is reported as such, not as a generic 'finished'.
            self.status = worker_state
        else:
            self.status = ('stopped' if self.control.get('action') == 'pause' and
                           self.control.get('resume_authorized') is not True
                           else 'finished')
        return dict(contract=self.contract, resources=self.resources,
                    status=self.status, control=self.control,
                    worker=self.worker_result)

    # ---------------------------------------------------------------- control
    def _control(self, action, **extra):
        self.control = dict(action=action, resume_authorized=False,
                            updated=float(self.clock()), **extra)
        return self.control

    # ---------------------------------------------------------------- probe
    def _write_probe(self, output, record):
        output.mkdir(parents=True, exist_ok=True)
        body = {key: value for key, value in record.items() if key != 'probe_sha256'}
        payload = dict(record, probe_sha256=hashlib.sha256(
            json.dumps(body, sort_keys=True, default=str).encode()).hexdigest())
        path = output / 'probe.json'
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + '\n',
                        encoding='utf-8')
        return payload, path

    def _execute_probe(self, output):
        """SDS 9.3 resource + recovery probe (engineering evidence only)."""
        plan = dict(self.contract['probe'])
        limits, _source = _resource_limits()
        started = float(self.clock())
        snapshot = self.resource_probe(directory=output.parent)
        if snapshot.get('process_mib') is None \
                and snapshot.get('process_mib_method') is None:
            snapshot['process_mib_method'] = 'not_initialized'
        recorder = dict(plan=plan, contract=self.contract, started=started,
                        output_dir=str(output), state='running', status='running',
                        resources=dict(limits=limits, pre_run=snapshot),
                        cost=dict(plan['cost']), recovery=None,
                        phase_timeline=[dict(stage='preflight', update=0,
                                             at=started)],
                        container=plan['container'], resumable=False,
                        acceptance_scope=PROBE_ACCEPTANCE_SCOPE,
                        counts_against_training=False,
                        training_budget_untouched=int(plan['training_budget_untouched']),
                        error_type=None)
        output.mkdir(parents=True, exist_ok=False)
        (output / 'preflight.json').write_text(
            json.dumps(recorder, indent=2, sort_keys=True, default=str) + '\n',
            encoding='utf-8')
        registry = self.gpu_registry or (self.preflight_dir / 'gpu_worker.json')
        acquire_single_worker(self.contract['device'], path=registry)
        try:
            pre = check_resources(snapshot, limits, stage='pre_run')
            recorder['resources']['pre_run_check'] = pre
            if not pre['ok']:
                recorder['state'] = 'resource_insufficient'
                recorder['status'] = 'resource_insufficient'
                recorder['process_mib_method'] = snapshot.get('process_mib_method')
                recorder['resources']['blocking'] = pre['blocking']
                recorder['phase_timeline'].append(dict(
                    stage='aborted', update=0, at=float(self.clock()),
                    reason='resource_insufficient'))
                payload, path = self._write_probe(output, recorder)
                self.status = 'resource_insufficient'
                self.worker_result = dict(state='resource_insufficient',
                                          probe=payload, path=str(path))
                return dict(contract=self.contract, resources=recorder['resources'],
                            status='resource_insufficient', control=dict(self.control),
                            probe=payload, path=str(path))
            if self.probe_factory is None:
                raise AdmissionError('no probe worker is wired; the supervisor cannot '
                                     'run the resource/recovery probe')
            result = self.probe_factory(plan, output)
            if not isinstance(result, dict):
                raise ProbeError('the probe worker returned no result mapping')
            cost = result.get('cost') or {}
            for key, value in plan['cost'].items():
                if int(cost.get(key, -1)) != int(value):
                    raise ProbeError(f'probe cost {key!r} differs from the frozen plan')
            validate_probe_recovery(result.get('recovery'), plan)
            samples = (result.get('resources') or {}).get('samples') or []
            summary = summarize_resource_samples(samples, limits)
            recorder.update(
                cost=dict(plan['cost']), recovery=dict(result['recovery']),
                resources=dict(recorder['resources'], measured=summary),
                phase_timeline=list(result.get('phase_timeline') or []),
                checkpoint=result.get('checkpoint'),
                checkpoint_sha256=result.get('checkpoint_sha256'),
                counts=dict(result.get('counts') or probe_ledger(plan)),
                finished=float(self.clock()))
            recorder['process_mib_method'] = (
                summary['metrics'].get('process_mib', {}).get('method')
                or recorder['resources'].get('pre_run', {}).get('process_mib_method'))
            if not summary['ok']:
                recorder['state'] = 'resource_insufficient'
                recorder['status'] = 'resource_insufficient'
                recorder['resources']['blocking'] = summary['blocking']
            else:
                recorder['state'] = 'engineering_check_only'
                recorder['status'] = 'engineering_check_only'
            payload, path = self._write_probe(output, recorder)
            self.status = recorder['state']
            self.worker_result = dict(state=recorder['state'], probe=payload,
                                      path=str(path))
            return dict(contract=self.contract, resources=recorder['resources'],
                        status=recorder['state'], control=dict(self.control),
                        probe=payload, path=str(path))
        except BaseException as error:
            recorder['state'] = 'error'
            recorder['status'] = 'error'
            recorder['error_type'] = type(error).__name__
            recorder['error'] = repr(error)
            recorder['finished'] = float(self.clock())
            self._write_probe(output, recorder)
            (output / 'error.json').write_text(
                json.dumps(dict(error_type=type(error).__name__, error=repr(error)),
                           indent=2, sort_keys=True) + '\n', encoding='utf-8')
            self.status = 'stopped'
            raise
        finally:
            release_single_worker(registry)

    def read_control(self, output=None):
        output = Path(output or self.contract['output_dir'])
        from .operator_control import ControlProtocol
        request=ControlProtocol(output/'control.json',output/'control_evidence').poll()
        return request or dict(action='continue',resume_authorized=False)

    def request_stop(self, reason='user_requested_stop', output=None):
        """Record an operator stop using the frozen spec state names (Q002-4)."""
        output = Path(output or (self.contract or {}).get('output_dir', '.'))
        spec = spec_module()
        allowed = tuple(getattr(spec, 'STOP_STATES', ()))
        if reason!='user_requested_stop':
            raise AdmissionError('operator controls may only request user_requested_stop')
        action = 'stop' if reason == 'user_requested_stop' else str(reason)
        record = dict(action=action, resume_authorized=False,
                      updated=float(self.clock()), reason=reason,
                      stop_state=reason,
                      failure_class=self._failure_class(reason))
        from .operator_control import write_control
        write_control(output/'control.json',record)
        self.control = record
        return record

    @staticmethod
    def _failure_class(state):
        from .runner import classify_failure
        return classify_failure(state)

    def state(self):
        return dict(status=self.status, contract=self.contract,
                    resources=self.resources, control=dict(self.control),
                    worker=self.worker_result)


def main(argv=None, *, worker_factory=None, resource_probe=None,
         probe_factory=None, root=None):
    """Entry point used by ``scripts/run_a2_ideal_stairs_v07.py``.

    ``--mode probe`` uses ``probe_factory`` exclusively; a missing probe worker is
    a clear error rather than a silent fallback to the training worker.
    """
    args = build_parser().parse_args(argv)
    orchestrator = V07Orchestrator(args, root=root,
                                   worker_factory=worker_factory,
                                   probe_factory=probe_factory,
                                   resource_probe=resource_probe)
    record = orchestrator.preflight()
    print(json.dumps(dict(mode=record['mode'], execute=record['execute'],
                          preflight_only=record['preflight_only'],
                          output_dir=record['output_dir'],
                          output_directory_created=record['output_directory_created'],
                          path=record['path']), indent=2, sort_keys=True))
    if not record['execute']:
        return 0
    result = orchestrator.execute()
    print(json.dumps(dict(status=result['status'],
                          control=result['control']), indent=2, sort_keys=True))
    return 0


__all__ = [
    'AdmissionError', 'DEFAULT_PREFLIGHT_DIR', 'PHASE_BUDGET_ORDER', 'PreflightError',
    'V07Orchestrator', 'acquire_single_worker', 'build_parser', 'check_resources',
    'command_ranges', 'expected_phase_budgets', 'main', 'parse_phase_budgets',
    'pid_alive', 'release_single_worker', 'resource_limits', 'resource_snapshot',
    'spec_module', 'validate_arguments', 'worker_registry_path',
]
