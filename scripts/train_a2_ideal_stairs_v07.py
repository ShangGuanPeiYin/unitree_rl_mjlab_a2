"""Native v07 worker adapters and separately billed engineering probe.

Production admission and the unique stop4817 continuation belong to runtime.py.
This module retains the C1 environment/case routing/PPO worker call paths and
never implements a fresh production initializer.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
for entry in (ROOT,ROOT/'scripts'):
    if str(entry) not in sys.path:sys.path.insert(0,str(entry))
import a2_v07_bootstrap
a2_v07_bootstrap.activate(ROOT)

def wandb_enabled():
    """W&B is enabled by an explicit A2_V07_WANDB flag or a present API key.

    ``A2_V07_WANDB=0/false/no/off`` always wins, so CPU/offline runs stay offline.
    """
    flag = str(os.environ.get('A2_V07_WANDB', '')).strip().lower()
    if flag in ('1', 'true', 'yes', 'on'):
        return True
    if flag in ('0', 'false', 'no', 'off'):
        return False
    return bool(os.environ.get('WANDB_API_KEY'))


def build_telemetry(output_dir, seed, *, backend=None):
    """Reliable W&B connection when enabled, plain local telemetry otherwise.

    Both classes share the ``V07Telemetry`` interface (start/update/gate/retention/
    finish), so the runner/recovery code never branches on the transport.  The
    reliable transport keeps the existing 2 s retry / 30 s heartbeat / 120 s
    no-progress / 32 in-flight policy (untouched, owned by ``connection.py``).
    """
    from src.tasks.velocity.v07.telemetry import V07Telemetry, WandbBackend

    directory = Path(output_dir) / 'telemetry'
    enabled = backend is not None or wandb_enabled()
    if not enabled:
        return V07Telemetry(directory)
    from src.tasks.velocity.v07.connection import ReliableConnection

    if backend is None:
        backend = WandbBackend(name=f'a2-v07-seed{seed}')
    return ReliableConnection(directory=directory, backend=backend)


def build_environment(contract, phase='F0', substage=None, evaluation=False,
                      num_envs=None, fixed_vx=None, initial_cases=None,
                      level_tick=0, frontier_tile_id=None):
    """Construct the native v07 environment through its unchanged lifecycle API."""
    from src.tasks.velocity.v07.env import V07Env  # imported lazily
    return V07Env(num_envs=int(num_envs or contract['num_envs']),
                  device=contract['device'], seed=int(contract['seed']),
                  phase=str(phase), substage=substage, evaluation=evaluation,
                  fixed_vx=fixed_vx, initial_cases=initial_cases,
                  level_tick=int(level_tick or 0),
                  frontier_tile_id=frontier_tile_id)


def case_environment_kwargs(contract, cases, phase, substage):
    """Route each batch by its frozen case identity, never the startup closure.

    Retention keeps its training-origin phase as provenance while measuring
    F0/H00. D/L physical substages come from the declared target terrain.
    """
    import math
    from numbers import Integral
    from src.tasks.velocity.v07 import evaluation, spec

    batch = list(cases)
    if not batch or len(batch) > int(contract['eval_num_envs']):
        raise ValueError('an evaluation batch must contain 1..eval_num_envs cases')
    identities = []
    for case in batch:
        if not isinstance(case, dict):
            raise ValueError('each evaluation case must be a mapping')
        tile = case.get('tile_id')
        if isinstance(tile, bool) or not isinstance(tile, Integral):
            raise ValueError('each case needs an integer target tile')
        tile = int(tile)
        try:
            terrain = spec.terrain_spec(tile)
        except (KeyError, ValueError, TypeError) as error:
            raise ValueError('evaluation target tile is unknown') from error
        origin_phase = case.get('phase')
        if not isinstance(origin_phase, str) or origin_phase not in spec.PHASE_ORDER:
            raise ValueError('each case needs its explicit valid phase')
        kind = case.get('kind')
        if kind not in evaluation.SEED_KIND:
            raise ValueError('each case needs its explicit valid evaluation kind')
        if kind == 'test' and origin_phase != 'R':
            raise ValueError('final evaluation cases must belong to R')
        if case.get('lesson_id') != terrain.lesson_id:
            raise ValueError('case lesson disagrees with its target tile')
        measurement = evaluation.speed.measurement_stage(origin_phase, kind)
        if case.get('measurement_stage') != measurement:
            raise ValueError('case measurement stage disagrees with its protocol')
        physical_phase = ('F0' if kind == 'flat_retention' else
                          'H' if kind == 'stair_retention' else origin_phase)
        if physical_phase in ('D', 'L'):
            allowed = spec.D_SUBSTAGES if physical_phase == 'D' else spec.L_SUBSTAGES
            if terrain.lesson_id not in allowed:
                raise ValueError('target lesson does not belong to the physical phase')
            physical_substage = terrain.lesson_id
        else:
            physical_substage = None
        if 'substage' in case and case['substage'] != physical_substage:
            raise ValueError('explicit case substage conflicts with its physical target')
        if physical_phase == 'F0' and tile != int(spec.FLAT_TILE_ID):
            raise ValueError('flat evaluation must measure the flat tile')
        if physical_phase == 'H' and not spec.tile_id_for('H00') <= tile <= spec.tile_id_for('H15'):
            raise ValueError('H evaluation must measure an H terrain')
        if kind == 'stair_retention' and tile != spec.tile_id_for('H00'):
            raise ValueError('stair retention must measure H00')
        if physical_phase in ('V', 'R') and tile != int(spec.FINAL_TILE_ID):
            raise ValueError('V/R evaluation must measure L40')
        raw_vx = case.get('vx')
        if isinstance(raw_vx, bool) or not isinstance(raw_vx, (int, float)) or not math.isfinite(float(raw_vx)):
            raise ValueError('case speed must be finite')
        vx = float(raw_vx)
        protocol_phase = physical_phase if kind in ('flat_retention', 'stair_retention') else origin_phase
        if not any(abs(vx-float(value)) < evaluation.speed.SPEED_TOLERANCE
                   for value in evaluation.speed.gate_speeds(protocol_phase)):
            raise ValueError('case speed is outside its frozen evaluation protocol')
        identities.append((origin_phase, kind, measurement, tile, terrain.lesson_id,
                           vx, physical_phase, physical_substage,
                           int(case.get('level_tick', 0) or 0)))
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError('evaluation cases disagree on the batch protocol or physical target')
    _, _, _, tile, _, vx, physical_phase, physical_substage, level_tick = identities[0]
    return dict(contract=contract, phase=physical_phase, substage=physical_substage,
                evaluation=True, num_envs=len(batch), fixed_vx=vx,
                initial_cases=batch, level_tick=level_tick, frontier_tile_id=tile)


def build_case_runner(model, contract, phase, substage):
    """Deterministic-mean case runner for gates, retention and the final test."""
    from src.tasks.velocity.v07.runner import environment_case_runner, spec_tile

    def factory(cases):
        # One environment per distinct case in the batch (SDS 8 evaluation envs).
        kwargs = case_environment_kwargs(contract, cases, phase, substage)
        contract_kwargs = dict(kwargs)
        nested = contract_kwargs.pop('contract')
        return build_environment(nested, **contract_kwargs)

    return environment_case_runner(model, factory, device=contract['device'])


def build_measured_resource_guard(output_dir, device, holder):
    """Fail-closed post-start resource sample (SDS 8.5, measured window).

    The training process owns a GPU allocation only after CUDA initialization and
    the first complete update, so the pre-run admission exempts ``process_mib``;
    this guard samples it once at the first update boundary and then returns the
    cached verdict.  A missing or over-5120-MiB process measurement stops the run
    as ``resource_insufficient`` (never silently passes).
    """
    from src.tasks.velocity.v07 import orchestration as orchestration_module

    resources_dir = Path(output_dir) / 'resources'
    cache = {}
    index = 0
    try:
        index = int(str(device).split(':')[-1]) if ':' in str(device) else 0
    except ValueError:
        index = 0

    def guard():
        if 'result' in cache:
            return cache['result']
        snapshot = orchestration_module.resource_snapshot(
            directory=Path(output_dir).parent, device=index)
        check = orchestration_module.check_resources(snapshot, stage='measured')
        update = None
        if holder.get('runner') is not None:
            update = int(holder['runner'].update)
        record = dict(stage='measured', update=update, check=check,
                      snapshot=snapshot, exempted=check['exempted'],
                      process_mib_method=snapshot.get('process_mib_method'))
        resources_dir.mkdir(parents=True, exist_ok=True)
        path = resources_dir / 'measured_first_update.json'
        path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + '\n',
                        encoding='utf-8')
        cache['result'] = dict(ok=check['ok'], blocking=check['blocking'],
                               checks=check['checks'], stage='measured',
                               snapshot=snapshot, update=update, path=str(path),
                               process_mib_method=snapshot.get('process_mib_method'))
        return cache['result']

    return guard


def probe_counts(checkpoint_learning, after_learning, plan):
    """The cumulative probe counts: checkpoint (55) + the single recovery update.

    The checkpoint ``learning`` block is authoritative for the 55 billed updates;
    the resumed runner's block is cumulative 56.  A missing cumulative field, a
    wrong single-update increment or a total that differs from the frozen plan is
    a `ProbeError` (never silently accepted).
    """
    from src.tasks.velocity.v07 import orchestration as orchestration_module
    from src.tasks.velocity.v07 import ppo as ppo_module

    required = ('update', 'control_samples', 'physics_substeps', 'adam_steps')
    blocks = dict(checkpoint_learning=checkpoint_learning, after_learning=after_learning)
    for label, block in blocks.items():
        if not isinstance(block, dict):
            raise orchestration_module.ProbeError(f'{label} is not a learning block')
        missing = [key for key in required if key not in block]
        if missing:
            raise orchestration_module.ProbeError(
                f'{label} is missing the cumulative probe counters {missing}; the probe '
                'checkpoint must save them')
    before = int(checkpoint_learning['update'])
    after = int(after_learning['update'])
    cost = dict(plan['cost'])
    if after - before != 1:
        raise orchestration_module.ProbeError(
            f'the probe recovery billed {after - before} updates; exactly one is required')
    per_update = dict(
        control_samples=int(plan['num_envs']) * int(plan['rollout']),
        physics_substeps=ppo_module.physics_substeps(
            int(plan['num_envs']) * int(plan['rollout'])),
        adam_updates=int(plan['epochs']) * int(plan['minibatches']))
    delta = dict(
        control_samples=int(after_learning['control_samples']) - int(
            checkpoint_learning['control_samples']),
        physics_substeps=int(after_learning['physics_substeps']) - int(
            checkpoint_learning['physics_substeps']),
        adam_updates=int(after_learning['adam_steps']) - int(
            checkpoint_learning['adam_steps']))
    if delta != per_update:
        raise orchestration_module.ProbeError(
            f'the single recovery update billed {delta}, expected {per_update}')
    counts = dict(
        learning_updates=after,
        control_samples=int(after_learning['control_samples']),
        physics_substeps=int(after_learning['physics_substeps']),
        adam_updates=int(after_learning['adam_steps']),
        update=after,
        optimizer_step=int(after_learning.get('optimizer_step', 0)),
        counts_against_training=False,
        training_budget_untouched=int(plan['training_budget_untouched']),
        training_budget_consumed=0,
        container=plan['container'], resumable=False)
    expected = dict(
        learning_updates=int(cost['learning_updates']),
        control_samples=int(cost['control_samples']),
        physics_substeps=int(cost['physics_substeps']),
        adam_updates=int(cost['adam_updates']))
    actual = {key: int(counts[key]) for key in expected}
    if actual != expected:
        raise orchestration_module.ProbeError(
            f'the cumulative probe counts {actual} do not reach the frozen plan '
            f'{expected}')
    return counts, dict(delta=delta, before=before, after=after,
                        per_update=per_update)


def probe_curriculum(plan, seed):
    """Explicit R/L40 curriculum state for the probe (never a fresh F0 manager)."""
    from src.tasks.velocity.v07 import spec as spec_module
    from src.tasks.velocity.v07.curriculum import CurriculumManager

    manager = CurriculumManager(num_envs=int(plan['num_envs']), seed=int(seed))
    state = manager.state
    state.phase = 'R'
    state.substage = None
    state.frontier_tile_id = int(spec_module.FINAL_TILE_ID)
    state.generation = 0
    state.generation_updates = 0
    state.last_update = 0
    state.level_tick = 0
    state.spent_by_phase = {name: 0 for name in spec_module.PHASE_ORDER}
    state.spent_by_substage = {}
    state.finished = False
    state.stop_status = None
    state.retention_enabled = {'flat': False, 'h00': False}
    state.retention_qualified_at = {'flat': None, 'h00': None}
    state.retention_next_due = {'flat': None, 'h00': None}
    state.retention_last_completed = {'flat': None, 'h00': None}
    state.retention_streaks = {'flat': 0, 'h00': 0}
    state.gate_log = []
    manager.load_state_dict(manager.state_dict())
    return manager


def build_probe_worker(plan, output_dir):
    """SDS 6.2 probe: 512 train envs + 8 eval envs on L40, update50 diagnostic,
    save55/resume56, exactly one non-replayed update, plus resource samples.

    The 56 probe updates are billed to the probe ledger only: they are returned
    with ``counts_against_training=False`` and never touch the 48,000 budget.
    """
    import time

    from src.tasks.velocity.v07 import model as model_module
    from src.tasks.velocity.v07 import orchestration as orchestration_module
    from src.tasks.velocity.v07 import ppo as ppo_module
    from src.tasks.velocity.v07 import recovery as recovery_module
    from src.tasks.velocity.v07 import runner as runner_module
    from src.tasks.velocity.v07.curriculum import CurriculumManager

    output_dir = Path(output_dir)
    device = str(plan.get('device') or 'cuda:0')
    seed = int(plan['seed'])
    contract = dict(num_envs=int(plan['num_envs']),
                    eval_num_envs=int(plan['eval_num_envs']),
                    device=device, seed=seed)
    timeline, samples, evaluation_resources = [], [], []

    def sample(update):
        snapshot = orchestration_module.resource_snapshot(
            directory=output_dir.parent, device=0)
        samples.append(dict(update=int(update),
                            whole_device_mib=snapshot.get('whole_device_mib'),
                            process_mib=snapshot.get('process_mib'),
                            process_mib_method=snapshot.get('process_mib_method'),
                            host_available_mib=snapshot.get('host_available_mib'),
                            disk_free_gib=snapshot.get('disk_free_gib')))

    def build(directory, mode, *, phase='R', curriculum_state=None, container=None,
              frontier_tile_id=None):
        model = (model_module.V07ActorCritic.migrate_eth_checkpoint(
            seed=seed, device=device) if mode == 'train'
            else model_module.V07ActorCritic().to(device))
        env = build_environment(contract, phase=phase,
                                frontier_tile_id=frontier_tile_id)
        trainer = ppo_module.PPOTrainer(model, env)
        # The first probe runner uses mode='train' for its own billing; the probe
        # terrain must still be the explicit R/L40 state (never a fresh F0 manager).
        if curriculum_state is None:
            manager = probe_curriculum(plan, seed)
        else:
            manager = CurriculumManager(num_envs=int(plan['num_envs']), seed=seed,
                                        state=curriculum_state)
        instance = runner_module.V07Runner(
            model=model, ppo=trainer, curriculum=manager, env=env,
            directory=directory, seed=seed, device=device, mode=mode,
            container=container or plan['container'], evaluate=True,
            autosave=False, require_resume_evidence=(mode == 'probe'),
            # Both probe instances write the probe container; the source mode stays
            # 'train' so the probe's own formal recovery reader can re-read them.
            source=recovery_module.source_block(seed=seed, device=device,
                                                mode='train'))
        instance.case_runner = probe_case_runner(
            build_case_runner(model, contract, phase, None), output_dir,
            evaluation_resources, lambda: int(instance.update))
        return instance

    from src.tasks.velocity.v07 import spec as spec_module

    runner = build(output_dir / 'train', 'train', phase='R',
                   container=plan['container'],
                   frontier_tile_id=int(spec_module.FINAL_TILE_ID))
    stages = plan['stages']
    for index in range(stages['warmup'] + stages['measured']):
        runner.train_one_update()
        if int(runner.update) == 50:
            capture_probe50_boundary(runner, output_dir)
        stage = 'warmup' if index < stages['warmup'] else 'measured'
        timeline.append(dict(stage=stage, update=int(runner.update),
                             at=float(time.time())))
        if stage == 'measured':
            sample(runner.update)
    diagnostic_evidence = probe_diagnostic_evidence(runner)
    checkpoint = runner.save('probe')
    state = recovery_module.preload_checkpoint(
        checkpoint['path'], checkpoint['sha256'], mode='probe')
    resumed = build(output_dir / 'resume', 'probe',
                    phase=str(state['curriculum']['phase']),
                    curriculum_state=state['curriculum'],
                    frontier_tile_id=int(state['curriculum'].get(
                        'frontier_tile_id', spec_module.FINAL_TILE_ID)))
    resume_contract = resumed.load(checkpoint['path'], checkpoint['sha256'],
                                   allow_probe=True)
    # Q006 section 6/probe: the production bare-model resume entry must persist and
    # formally reload-verify the loaded state at 55 before update 56, and again
    # after 56.  The existing metadata object is never reused as a stand-in.
    loaded = resumed.save_verified('resume_loaded', allow_probe=True)
    checkpoint_learning = dict(state['learning'])
    resumed.train_one_update()
    after_learning = dict(resumed.learning)
    ready = resumed.save_verified('resume_ready', allow_probe=True)
    counts, count_evidence = probe_counts(checkpoint_learning, after_learning, plan)
    timeline.append(dict(stage='recovery', update=int(counts['update']),
                         at=float(time.time())))
    receipt = dict(
        resume_mode=resume_contract['resume_mode'],
        resumed_from_update=count_evidence['before'],
        updates_after=count_evidence['after'] - count_evidence['before'],
        billed_updates_not_replayed=bool(
            count_evidence['before'] == plan['cost']['learning_updates'] - 1
            and count_evidence['after'] == plan['cost']['learning_updates']),
        initialized_from_eth_again=False,
        container=state.get('container'),
        checkpoint_sha256=checkpoint['sha256'],
        curriculum_state=resumed._curriculum_state(),
        resume_loaded=dict(path=loaded['path'], sha256=loaded['sha256'],
                           update=int(loaded['update']),
                           verification=dict(loaded['verification'])),
        resume_ready=dict(path=ready['path'], sha256=ready['sha256'],
                          update=int(ready['update']),
                          verification=dict(ready['verification'])),
        counts=counts,
        counts_at_checkpoint=checkpoint_learning,
        single_update_delta=count_evidence['delta'])
    return dict(cost=dict(plan['cost']), resources=dict(samples=samples, evaluation_batches=evaluation_resources),
                recovery=receipt, phase_timeline=timeline,
                checkpoint=checkpoint['path'],
                checkpoint_sha256=checkpoint['sha256'], counts=counts,
                diagnostic_update50=diagnostic_evidence)



def probe_case_runner(base_runner, output_dir, evidence_rows, current_update):
    """Measure every real diagnostic batch without changing the case runner."""
    from validate_a2_ideal_stairs_v07 import resource_after_batch, evaluation_cost
    counter=0
    def run(cases):
        nonlocal counter
        records=base_runner(cases)
        counter+=1
        label=f'probe_update{int(current_update()):06d}_batch{len(evidence_rows)+counter:03d}'
        resource=resource_after_batch(output_dir,label)
        path=Path(output_dir)/'resources'/(label+'.json')
        from src.tasks.velocity.v07 import spec
        evidence_rows.append(dict(update=int(current_update()),cost=evaluation_cost(records),
            resource_binding=dict(path=str(path),sha256=spec.sha256(path)),resource=resource))
        return records
    run.supports_batch=True
    return run


def capture_probe50_boundary(runner, output_dir):
    """Observe live model/Adam/task immediately after PPO50, before PPO51.

    This independent observer never reads a sidecar's identity as its expected
    identity. The returned file binding is retained separately until update55.
    """
    from copy import deepcopy
    from src.tasks.velocity.v07 import evaluation, model, recovery, spec, orchestration
    from validate_a2_ideal_stairs_v07 import write_json
    if (int(runner.update), runner.phase, runner.substage, int(runner.seed)) != (50, 'R', None, 42):
        raise orchestration.ProbeError('probe observer must capture the live R boundary50')
    learning=deepcopy(runner.ppo.learning)
    expected_learning=dict(update=50, control_samples=50*spec.CONTROL_SAMPLES_PER_UPDATE,
        physics_substeps=50*spec.PHYSICS_SUBSTEPS_PER_UPDATE,
        adam_steps=50*spec.ADAM_UPDATES_PER_ITERATION)
    if any(learning.get(key)!=value for key,value in expected_learning.items()):
        raise orchestration.ProbeError('probe observer learning counters are not boundary50')
    source_sha=recovery.digest(spec.source_hashes())
    if runner.source.get('source_closure_sha256')!=source_sha:
        raise orchestration.ProbeError('probe observer source differs from the live frozen closure')
    table=evaluation.make_cases('R',None,'diagnostic',42,tile_id=spec.FINAL_TILE_ID)
    optimizer_state=runner.ppo.optimizer.state_dict()
    if not optimizer_state.get('state'):
        raise orchestration.ProbeError('probe observer must capture the actual nonempty Adam state')
    expected=dict(update=50,phase='R',substage=None,
        generation=runner.curriculum.state_dict()['generation'],kind='diagnostic',
        case_table_sha256=table.sha256,round=table.round,
        model_sha256=model.state_sha256(runner.model.state_dict()),
        optimizer_sha256=recovery.digest(evaluation._optimizer_digest(optimizer_state)),
        source_closure_sha256=source_sha,contract_sha256=spec.contract_sha256())
    rows=[row for row in runner.diagnostics if int(row.get('update',-1))==50]
    if len(rows)!=1:
        raise orchestration.ProbeError('probe observer lacks its committed diagnostic50 entry')
    payload=dict(schema='v07-probe-live-boundary50-v1',identity=expected,
        learning=learning,diagnostic_entry=deepcopy(rows[0]))
    binding=write_json(Path(output_dir)/'probe_boundary50.json',payload)
    runner.probe50_boundary=deepcopy(binding)
    return binding


def _probe_first_terminal_records(result,expected_identity):
    """Cross-check actual first-terminal exports against their cases and slots."""
    import torch
    from src.tasks.velocity.v07 import runner as production, evaluation
    rows=result.records
    if [row.get('case_id') for row in rows]!=[case['case_id'] for case in result.table.cases]:
        raise evaluation.EvaluationError('probe diagnostic case order differs from the frozen task')
    position=0
    for batch in evaluation.batch_cases(result.table):
        batch_rows=rows[position:position+len(batch)];position+=len(batch)
        loop_steps=set()
        for index,(case,row) in enumerate(zip(batch,batch_rows)):
            raw=row.get('first_episode')
            if not isinstance(raw,dict) or row.get('evidence_kind')!='terminal':
                raise evaluation.EvaluationError('probe diagnostic requires immutable first-terminal evidence')
            if (type(row.get('env_index')) is not int or row['env_index']!=index
                or type(raw.get('env_id')) is not int or raw['env_id']!=index):
                raise evaluation.EvaluationError('probe first-terminal case/env slot mismatch')
            step=production._first_case_integer(row.get('steps'),'probe case steps',minimum=1)
            observed=production._first_case_integer(row.get('first_observed_at_batch_step'),
                'probe first observed step',minimum=1)
            production._validate_first_case(raw,index=index,batch_step=observed,
                evidence_kind='terminal',flat=False)
            if step!=observed or raw.get('control_samples')!=step or raw.get('terminal_step')!=step:
                raise evaluation.EvaluationError('probe first-terminal step identity differs')
            if (raw.get('tile_id')!=case['tile_id'] or raw.get('lesson_id')!=case['lesson_id']
                or raw.get('frontier_tile_id')!=case['tile_id']
                or raw.get('phase')!=case['phase'] or raw.get('substage') is not None
                or raw.get('seed')!=result.table.seed
                or raw.get('generation')!=expected_identity['generation']):
                raise evaluation.EvaluationError('probe first-terminal physical case identity differs')
            target=[float(value) for value in torch.tensor(case['initial_pose'],dtype=torch.float32)]
            pose=raw.get('initial_pose')
            if (not isinstance(pose,list) or len(pose)!=3
                or not all(production._same_case_scalar(got,want) for got,want in zip(pose,target))):
                raise evaluation.EvaluationError('probe first-terminal initial pose differs from its case')
            command=float(torch.tensor(case['vx'],dtype=torch.float32))
            if not production._same_case_scalar(raw.get('command'),command):
                raise evaluation.EvaluationError('probe first-terminal executed command differs from its case')
            from src.tasks.velocity.v07 import events
            if ((raw['success'] and raw['reason']!='success')
                or (raw['timeout'] and raw['reason']!='timeout')
                or (raw['failure'] and raw['reason'] not in events.FAILURE_NAMES)):
                raise evaluation.EvaluationError('probe terminal reason disagrees with its outcome')
            outcomes=[raw[name] for name in ('success','failure','timeout')]
            if sum(outcomes)!=1:
                raise evaluation.EvaluationError('probe first-terminal outcomes are not mutually exclusive')
            for exported,source in [('climb_success','success'),('failure','failure'),('timeout','timeout'),
                ('reason','reason'),('active_samples','active_samples'),('stall_samples','stall_samples'),
                ('first_step_valid','first_step_valid'),('mean_vx','mean_vx'),('mean_speed','mean_speed'),
                ('max_abs_y','max_abs_y'),('max_abs_yaw_deg','max_abs_yaw_deg')]:
                if row.get(exported)!=raw.get(source):
                    raise evaluation.EvaluationError('probe first-terminal export disagrees: '+exported)
            loops=production._first_case_integer(row.get('batch_loop_control_steps'),
                'probe physical batch loops',minimum=1)
            if loops<step or row.get('physical_control_samples')!=loops:
                raise evaluation.EvaluationError('probe first-terminal physical batch steps differ')
            loop_steps.add(loops)
        if len(loop_steps)!=1:
            raise evaluation.EvaluationError('probe rows disagree on actual batch simulation work')


def probe_diagnostic_evidence(runner):
    """Verify raw diagnostic50 against the separately captured live PPO50 state."""
    from src.tasks.velocity.v07 import evaluation_records, recovery, orchestration, spec
    observer_binding=getattr(runner,'probe50_boundary',None)
    if not isinstance(observer_binding,dict) or set(observer_binding)!={'path','sha256'}:
        raise orchestration.ProbeError('probe requires the independent live boundary50 observer')
    observer_path=Path(observer_binding['path'])
    if not observer_path.is_file() or spec.sha256(observer_path)!=observer_binding['sha256']:
        raise orchestration.ProbeError('probe live boundary50 observer file is missing or changed')
    observer=json.loads(observer_path.read_text())
    if observer.get('schema')!='v07-probe-live-boundary50-v1':
        raise orchestration.ProbeError('probe live boundary50 observer schema differs')
    expected=observer['identity']
    if (expected.get('update'),expected.get('phase'),expected.get('substage'),expected.get('kind'))!=(50,'R',None,'diagnostic'):
        raise orchestration.ProbeError('probe live observer is not diagnostic50')
    if (expected.get('source_closure_sha256')!=recovery.digest(spec.source_hashes())
        or expected.get('contract_sha256')!=spec.contract_sha256()):
        raise orchestration.ProbeError('probe live observer source/contract differs from the frozen worker')
    rows=[row for row in runner.diagnostics if int(row.get('update',-1))==50]
    if len(rows)!=1 or rows[0]!=observer.get('diagnostic_entry'):
        raise orchestration.ProbeError('probe committed diagnostic entry differs from its live50 capture')
    row=rows[0]
    if row.get('phase')!='R' or row.get('substage') is not None:
        raise orchestration.ProbeError('probe committed diagnostic context differs')
    binding=row.get('case_records') or {}
    path=Path(binding.get('path',''))
    if not path.is_file():
        raise orchestration.ProbeError('probe update50 diagnostic raw sidecar is missing')
    payload=json.loads(path.read_text())
    evaluation_records.verify_sidecar(binding,payload)
    result=evaluation_records.restore_result(payload,expected)
    if row.get('evidence')!=result.evidence_sha256:
        raise orchestration.ProbeError('probe committed diagnostic evidence differs from the actual raw result')
    if len(result.records)!=15 or result.table.target_tile_id!=spec.FINAL_TILE_ID:
        raise orchestration.ProbeError('probe update50 diagnostic count/terrain differs')
    _probe_first_terminal_records(result,expected)
    from validate_a2_ideal_stairs_v07 import evaluation_cost
    return dict(update=50,case_count=15,case_records=dict(binding),
        boundary50=dict(observer_binding),identity=dict(expected),
        evidence_sha256=dict(result.evidence_sha256),**evaluation_cost(result.records))


def build_worker(contract,output_dir):
    """Delegate unique migration/resume admission to the native runtime owner."""
    from src.tasks.velocity.v07.runtime import build_worker as execute
    return execute(contract,output_dir)


def main(argv=None):
    from src.tasks.velocity.v07.runtime import main as execute
    return execute(argv)


if __name__=='__main__':raise SystemExit(main())
