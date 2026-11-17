"""v0.6 training worker: build the single continuous F0->H->D->L->V->R run.

This entry point owns the *worker* half of the SDS 11.2 contract: it builds the
ETH-migrated network, the V06 environment, the single-Adam PPO trainer, the
curriculum manager and the schema-8 runner.  Admission (preflight, resources,
single GPU worker, directory refusal) is owned by
``src/tasks/velocity/v06/orchestration.py``; this module never re-implements a
CLI assertion.

``--execute`` is required to start training.  Without it the supervisor only runs
a preflight and prints the validated contract.

Usage::

    MUJOCO_GL=egl /home/xxs/anaconda3/envs/a2/bin/python \\
      scripts/train_a2_ideal_stairs_v06.py --mode train --execute --recipe 0.6-go2refs-r1 \\
      --seed 42 --device cuda:0 --num-envs 512 --eval-num-envs 8 --rollout 48 \\
      --epochs 5 --minibatches 4 --learning-rate 0.0003 --gamma 0.99 \\
      --entropy-coef 0.003 --initial-std 0.30 --std-min 0.05 --std-max 1.00 \\
      --initial-vx-min 0.30 --initial-vx-max 0.80 --final-vx-min 0.20 \\
      --final-vx-max 0.60 --vy 0.0 --wz 0.0 --max-updates 30000 \\
      --phase-budgets F0=1000,H=12000,D=3000,L=6000,V=4000,R=4000 \\
      --save-interval 100 --diagnostic-interval 50 --gate-interval 100 \\
      --init-checkpoint /home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt \\
      --init-sha256 8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722 \\
      --acceptance-file logs/a2_ideal_stairs_v06/implementation/V06_IMPLEMENTATION_ACCEPTANCE.md \\
      --output-dir logs/a2_ideal_stairs_v06/train_seed42_trial01
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / 'scripts'):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import a2_v06_bootstrap  # noqa: E402  (stdlib-only, before any src.tasks import)

a2_v06_bootstrap.activate(str(ROOT))


def wandb_enabled():
    """W&B is enabled by an explicit A2_V06_WANDB flag or a present API key.

    ``A2_V06_WANDB=0/false/no/off`` always wins, so CPU/offline runs stay offline.
    """
    flag = str(os.environ.get('A2_V06_WANDB', '')).strip().lower()
    if flag in ('1', 'true', 'yes', 'on'):
        return True
    if flag in ('0', 'false', 'no', 'off'):
        return False
    return bool(os.environ.get('WANDB_API_KEY'))


def build_telemetry(output_dir, seed, *, backend=None):
    """Reliable W&B connection when enabled, plain local telemetry otherwise.

    Both classes share the ``V06Telemetry`` interface (start/update/gate/retention/
    finish), so the runner/recovery code never branches on the transport.  The
    reliable transport keeps the existing 2 s retry / 30 s heartbeat / 120 s
    no-progress / 32 in-flight policy (untouched, owned by ``connection.py``).
    """
    from src.tasks.velocity.v06.telemetry import V06Telemetry, WandbBackend

    directory = Path(output_dir) / 'telemetry'
    enabled = backend is not None or wandb_enabled()
    if not enabled:
        return V06Telemetry(directory)
    from src.tasks.velocity.v06.connection import ReliableConnection

    if backend is None:
        backend = WandbBackend(name=f'a2-v06-seed{seed}')
    return ReliableConnection(directory=directory, backend=backend)


def build_environment(contract, phase='F0', substage=None, evaluation=False,
                      num_envs=None, fixed_vx=None, initial_cases=None,
                      level_tick=0, frontier_tile_id=None):
    """Construct the v0.6 environment (owned by agent B)."""
    from src.tasks.velocity.v06.env import V06Env  # imported lazily
    return V06Env(num_envs=int(num_envs or contract['num_envs']),
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
    from src.tasks.velocity.v06 import evaluation, spec

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
    from src.tasks.velocity.v06.runner import environment_case_runner, spec_tile

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
    from src.tasks.velocity.v06 import orchestration as orchestration_module

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


Q005_MANIFEST = (ROOT / 'logs/a2_ideal_stairs_v06/implementation/coordination/'
                 'Q005_recovery_bridge.md')
Q006_MANIFEST = (ROOT / 'logs/a2_ideal_stairs_v06/implementation/coordination/'
                 'Q006_recovery_bridge.md')
Q007_MANIFEST = (ROOT / 'logs/a2_ideal_stairs_v06/implementation/coordination/'
                 'Q007_recovery_bridge.md')


def verify_q005_bridge(contract, *, manifest=None):
    """Assert the runtime bridge identity against the recorded Q005 manifest."""
    from src.tasks.velocity.v06 import recovery as recovery_module

    path = Path(manifest or Q005_MANIFEST)
    if not path.is_file():
        raise SystemExit(f'Q005 recovery manifest is missing: {path}')
    text = path.read_text(encoding='utf-8')
    for label, value in (
            ('checkpoint', recovery_module.BRIDGE_CHECKPOINT_SHA256),
            ('old closure', recovery_module.BRIDGE_OLD_SOURCE_CLOSURE_SHA256),
            ('Q005', recovery_module.Q005_DECISION_SHA256)):
        if value not in text:
            raise SystemExit(f'Q005 manifest does not record the runtime {label} SHA')
    sha = recovery_module.sha256(path)
    resume_sha = (contract.get('resume') or {}).get('checkpoint_sha256')
    if resume_sha != recovery_module.BRIDGE_CHECKPOINT_SHA256:
        raise SystemExit('runtime resume checkpoint is not the documented Q005 checkpoint')
    return dict(manifest=str(path), manifest_sha256=sha,
                checkpoint_sha256=resume_sha,
                old_source_closure_sha256=(
                    recovery_module.BRIDGE_OLD_SOURCE_CLOSURE_SHA256),
                q005_sha256=recovery_module.Q005_DECISION_SHA256)


def verify_q006_bridge(contract, *, manifest=None):
    """Q006 section 4.6: the extended bridge binds Q005, Q006 and the errata."""
    from src.tasks.velocity.v06 import ledger as ledger_module
    from src.tasks.velocity.v06 import recovery as recovery_module

    path = Path(manifest or Q006_MANIFEST)
    if not path.is_file():
        raise SystemExit(f'Q006 recovery manifest is missing: {path}')
    text = path.read_text(encoding='utf-8')
    for label, value in (
            ('checkpoint', recovery_module.BRIDGE_CHECKPOINT_SHA256),
            ('old closure', recovery_module.BRIDGE_OLD_SOURCE_CLOSURE_SHA256),
            ('Q005', recovery_module.Q005_DECISION_SHA256),
            ('Q006', ledger_module.Q006_DECISION_SHA256),
            ('Q006 errata', ledger_module.Q006_ERRATA_SHA256)):
        if value not in text:
            raise SystemExit(f'Q006 manifest does not record the runtime {label} SHA')
    bridge = ((contract.get('resume') or {}).get('bridge') or {})
    segment = dict(bridge.get('segment') or {})
    if not bridge.get('q006_segment'):
        raise SystemExit('runtime resume did not confirm the Q006 segment bridge')
    if (int(segment.get('rollback_from_update', -1)),
            int(segment.get('resume_from_update', -1)),
            int(segment.get('lost_updates', -1))) != (150, 100, 50):
        raise SystemExit('Q006 segment metadata is not rollback150/resume100/lost50')
    if bridge.get('replay_billed_updates') is not True:
        raise SystemExit('Q006 bridge must declare replay_billed_updates=true')
    if (bridge.get('independent_acceptance_sha256')
            == ledger_module.Q006_ACCEPTANCE_SHA256):
        # The acceptance SHA is a documented placeholder until D fills it; it is
        # recorded as pending, never silently treated as a verified digest.
        pass
    return dict(manifest=str(path),
                manifest_sha256=recovery_module.sha256(path),
                checkpoint_sha256=recovery_module.BRIDGE_CHECKPOINT_SHA256,
                old_source_closure_sha256=(
                    recovery_module.BRIDGE_OLD_SOURCE_CLOSURE_SHA256),
                new_source_closure_sha256=bridge.get('new_source_closure_sha256'),
                q005_sha256=recovery_module.Q005_DECISION_SHA256,
                q006_sha256=ledger_module.Q006_DECISION_SHA256,
                q006_errata_sha256=ledger_module.Q006_ERRATA_SHA256,
                independent_acceptance_sha256=bridge.get(
                    'independent_acceptance_sha256'),
                acceptance_pending=(
                    bridge.get('independent_acceptance_sha256')
                    == ledger_module.Q006_ACCEPTANCE_SHA256),
                segment=segment,
                replay_billed_updates=True)


def verify_q007_bridge(contract, *, manifest=None):
    """Q007 section 4: bind the exception_000400 source and its single migration."""
    from src.tasks.velocity.v06 import ledger as ledger_module
    from src.tasks.velocity.v06 import recovery as recovery_module

    path = Path(manifest or Q007_MANIFEST)
    if not path.is_file():
        raise SystemExit(f'Q007 recovery manifest is missing: {path}')
    text = path.read_text(encoding='utf-8')
    for label, value in (
            ('source checkpoint', ledger_module.Q007_SOURCE_CHECKPOINT_SHA256),
            ('forbidden boundary',
             ledger_module.Q007_FORBIDDEN_CHECKPOINT_SHA256),
            ('old closure', ledger_module.Q007_OLD_SOURCE_CLOSURE_SHA256),
            ('Q006', ledger_module.Q006_DECISION_SHA256),
            ('Q007', ledger_module.Q007_DECISION_SHA256),
            ('Q007 errata', ledger_module.Q007_ERRATA_SHA256)):
        if value not in text:
            raise SystemExit(f'Q007 manifest does not record the runtime {label} SHA')
    bridge = ((contract.get('resume') or {}).get('bridge') or {})
    if not bridge.get('q007_segment'):
        raise SystemExit('runtime resume did not confirm the Q007 segment bridge')
    if bridge.get('checkpoint_sha256') != ledger_module.Q007_SOURCE_CHECKPOINT_SHA256:
        raise SystemExit('runtime resume source is not exception_000400.pt')
    segment = dict(bridge.get('segment') or {})
    if (int(segment.get('rollback_from_update', -1)),
            int(segment.get('resume_from_update', -1)),
            int(segment.get('lost_updates', -1))) != (400, 400, 0):
        raise SystemExit('Q007 segment metadata is not rollback400/resume400/lost0')
    if bridge.get('replay_billed_updates') is not False:
        raise SystemExit('Q007 must not replay billed updates')
    if bridge.get('replay_pending_evaluations') is not True:
        raise SystemExit('Q007 must replay exactly the migrated pending retention task')
    migration = bridge.get('task_migration')
    if not isinstance(migration, dict):
        raise SystemExit('Q007 bridge did not produce the pending-task migration plan')
    old_task = dict(migration.get('old_task') or {})
    new_task = dict(migration.get('new_task') or {})
    if (int(old_task.get('tile', -1)) != 1
            or old_task.get('case_table_sha256') !=
            ledger_module.Q007_OLD_FLAT_TASK_SHA256):
        raise SystemExit('Q007 old flat task is not the documented tile1/old table')
    if int(new_task.get('tile', -1)) != 0:
        raise SystemExit('Q007 new flat task does not measure the flat tile')
    if new_task.get('case_table_sha256') != migration.get('new_case_table_sha256'):
        raise SystemExit('Q007 new task SHA does not match the migrated table')
    if migration.get('plan_sha256') in (None, ''):
        raise SystemExit('Q007 migration plan has no digest')
    return dict(manifest=str(path),
                manifest_sha256=recovery_module.sha256(path),
                source_checkpoint_sha256=bridge.get('checkpoint_sha256'),
                forbidden_checkpoint_sha256=(
                    ledger_module.Q007_FORBIDDEN_CHECKPOINT_SHA256),
                old_source_closure_sha256=(
                    ledger_module.Q007_OLD_SOURCE_CLOSURE_SHA256),
                new_source_closure_sha256=bridge.get('new_source_closure_sha256'),
                q006_sha256=ledger_module.Q006_DECISION_SHA256,
                q007_sha256=ledger_module.Q007_DECISION_SHA256,
                q007_errata_sha256=ledger_module.Q007_ERRATA_SHA256,
                independent_acceptance_sha256=bridge.get(
                    'independent_acceptance_sha256'),
                acceptance_pending=(
                    bridge.get('independent_acceptance_sha256')
                    == ledger_module.Q007_ACCEPTANCE_SHA256),
                segment=segment,
                migration=dict(migration),
                replay_billed_updates=False,
                replay_pending_evaluations=True)


def build_worker(contract, output_dir):
    """Worker body called by the supervisor exactly once."""
    from src.tasks.velocity.v06 import model as model_module
    from src.tasks.velocity.v06 import ppo as ppo_module
    from src.tasks.velocity.v06 import recovery as recovery_module
    from src.tasks.velocity.v06 import runner as runner_module
    from src.tasks.velocity.v06 import telemetry as telemetry_module
    from src.tasks.velocity.v06.curriculum import CurriculumManager

    output_dir = Path(output_dir)
    device = contract['device']
    holder = {}
    resource_guard = build_measured_resource_guard(output_dir, device, holder)
    if contract['mode'] == 'resume':
        # Q005/Q006: the restricted bridge flag must reach the very first preload,
        # or a documented cross-source checkpoint is refused before the worker can
        # even read its own contract.
        bridge_allowed = bool(contract.get('allow_q005_bridge')
                              or contract.get('allow_q006_bridge')
                              or contract.get('allow_q007_bridge'))
        state = recovery_module.preload_checkpoint(
            contract['resume_run'], mode='resume', allow_bridge=bridge_allowed)
        model = model_module.V06ActorCritic().to(device)
        phase = str(state['curriculum']['phase']).upper()
        substage = state['curriculum'].get('substage')
        env = build_environment(contract, phase=phase, substage=substage)
        ppo = ppo_module.PPOTrainer(model, env)
        curriculum = CurriculumManager(num_envs=int(contract['num_envs']),
                                       seed=int(contract['seed']),
                                       state=state['curriculum'])
        allow_bridge = bridge_allowed
        evidence = None
        if contract.get('allow_q007_bridge'):
            evidence = verify_q007_bridge(contract)
        elif contract.get('allow_q006_bridge'):
            evidence = verify_q006_bridge(contract)
        elif allow_bridge:
            evidence = verify_q005_bridge(contract)
        runner = runner_module.V06Runner(
            model=model, ppo=ppo, curriculum=curriculum, env=env, directory=output_dir,
            seed=contract['seed'], device=device, mode='resume',
            telemetry=build_telemetry(output_dir, int(contract['seed'])),
            acceptance_file=contract['acceptance_file'],
            resource_guard=resource_guard, require_resume_evidence=True)
        holder['runner'] = runner
        # Q005/Q006 wiring: the resumed run must be able to complete its pending
        # boundary tasks (validation/diagnostic/retention) before update 101, so
        # the same V06Env-backed case runner used by the fresh path is installed
        # BEFORE load() replays the pending transaction.
        runner.case_runner = build_case_runner(model, contract, phase, substage)
        resumed = runner.load(contract['resume_run'], allow_bridge=allow_bridge)
        migration = (resumed.get('pending_task_migration')
                     if isinstance(resumed, dict) else None)
        if migration is not None:
            # Q007 section 4: the full old task / new task / new table content and
            # SHA / reason / decision SHA are appended to the migration manifest
            # before the flat set is executed.
            migrated_dir = output_dir / 'recovery_q007'
            migrated_dir.mkdir(parents=True, exist_ok=True)
            (migrated_dir / 'task_migration_000400.json').write_text(
                json.dumps(migration, indent=2, sort_keys=True, default=str) + '\n',
                encoding='utf-8')
            implementation = (ROOT / 'logs/a2_ideal_stairs_v06/implementation')
            implementation.mkdir(parents=True, exist_ok=True)
            manifest_path = implementation / 'Q007_task_migration.json'
            existing = []
            if manifest_path.is_file():
                existing = json.loads(manifest_path.read_text(encoding='utf-8'))
                if not isinstance(existing, list):
                    existing = [existing]
            if not any(isinstance(item, dict)
                       and item.get('plan_sha256') == migration.get('plan_sha256')
                       for item in existing):
                existing.append(dict(
                    target='train_seed42_trial02_resume_000400_q007',
                    plan_sha256=migration.get('plan_sha256'),
                    source=migration.get('source'),
                    pending=migration.get('pending'),
                    old_task=migration.get('old_task'),
                    old_task_protocol_invalid=migration.get(
                        'old_task_protocol_invalid'),
                    new_task=migration.get('new_task'),
                    new_case_table_sha256=migration.get('new_case_table_sha256'),
                    new_case_table=migration.get('new_case_table'),
                    completed_results=migration.get('completed_results'),
                    inherited=migration.get('inherited'),
                    reason=migration.get('reason'),
                    q007_decision_sha256=migration.get('q007_decision_sha256'),
                    q007_errata_sha256=migration.get('q007_errata_sha256'),
                    written_before_execution=True))
            manifest_path.write_text(
                json.dumps(existing, indent=2, sort_keys=True, default=str) + '\n',
                encoding='utf-8')
        # Q006/Q007 section 4.3: before any evaluation call (the pending boundary
        # replay itself) persist and formally reload-verify resume_loaded.
        loaded = runner.save_verified('resume_loaded')
        if float(runner.ppo.optimizer.param_groups[0]['lr']) \
                != float(resumed['learning_rate']):
            raise SystemExit('the bridge resume reset the checkpoint learning rate')
        receipt = dict(
            resume_mode=resumed['resume_mode'],
            checkpoint_sha256=resumed['checkpoint_sha256'],
            learning_rate=float(resumed['learning_rate']),
            pending_boundary=bool(resumed.get('pending_boundary_requires_completion')),
            replay_billed_updates=bool(resumed.get('replay_billed_updates')),
            segment=dict(resumed.get('segment') or {}),
            pending_task_migration=(
                None if resumed.get('pending_task_migration') is None else dict(
                    schema=resumed['pending_task_migration'].get('schema'),
                    plan_sha256=resumed['pending_task_migration'].get('plan_sha256'),
                    new_case_table_sha256=resumed['pending_task_migration'].get(
                        'new_case_table_sha256'))),
            resume_loaded=dict(path=loaded['path'], sha256=loaded['sha256'],
                               update=int(loaded['update']),
                               verification=dict(loaded['verification'])),
            transfer_metadata_deepcopied=True)
        if evidence is not None:
            evidence.update(receipt)
        if allow_bridge or evidence is not None:
            if contract.get('allow_q007_bridge'):
                recovery_dir = output_dir / 'recovery_q007'
            elif contract.get('allow_q006_bridge'):
                recovery_dir = output_dir / 'recovery_q006'
            else:
                recovery_dir = output_dir / 'recovery_q005'
            recovery_dir.mkdir(parents=True, exist_ok=True)
            (recovery_dir / 'receipt.json').write_text(
                json.dumps(evidence, indent=2, sort_keys=True, default=str) + '\n',
                encoding='utf-8')
            with (recovery_dir / 'history.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(evidence, sort_keys=True, default=str) + '\n')
    else:
        model = model_module.V06ActorCritic.migrate_eth_checkpoint(
            contract['init_checkpoint'], contract['init_sha256'],
            seed=int(contract['seed']), device=device)
        env = build_environment(contract, phase='F0')
        ppo = ppo_module.PPOTrainer(model, env)
        curriculum = CurriculumManager(num_envs=int(contract['num_envs']),
                                       seed=int(contract['seed']))
        telemetry = build_telemetry(output_dir, int(contract['seed']))
        runner = runner_module.V06Runner(
            model=model, ppo=ppo, curriculum=curriculum, env=env, directory=output_dir,
            seed=contract['seed'], device=device, mode='train', telemetry=telemetry,
            acceptance_file=contract['acceptance_file'],
            resource_guard=resource_guard)
        holder['runner'] = runner
        runner.case_runner = build_case_runner(model, contract, runner.phase,
                                               runner.substage)
        # Q004: a valid update-0 checkpoint exists before any first learning or
        # baseline evaluation can fail.
        runner.save('update0')
        telemetry.start(f'a2-v06-seed{contract["seed"]}', contract)
    telemetry = runner.telemetry
    result = runner.run(max_updates=int(contract['max_updates']))
    if contract['mode'] == 'resume' and runner.resume_ready is not None:
        # Q006 section 4.4: the completed update-100 boundary has its own verified
        # checkpoint before update 101; the receipt is an append-only record.
        if contract.get('allow_q007_bridge'):
            recovery_dir = output_dir / 'recovery_q007'
        elif contract.get('allow_q006_bridge'):
            recovery_dir = output_dir / 'recovery_q006'
        else:
            recovery_dir = output_dir / 'recovery_q005'
        recovery_dir.mkdir(parents=True, exist_ok=True)
        ready = runner.resume_ready
        (recovery_dir / 'resume_ready.json').write_text(
            json.dumps(dict(path=ready['path'], sha256=ready['sha256'],
                            update=int(ready['update']),
                            verification=dict(ready['verification']),
                            pending_boundary_cleared=bool(runner.pending_boundary is None),
                            resume_origin=runner.resume_origin,
                            first_new_update=int(runner.update) + 1),
                       indent=2, sort_keys=True, default=str) + '\n',
            encoding='utf-8')
    if result.get('state') == 'complete':
        result['final_test'] = runner.final_test()
    if telemetry is not None:
        telemetry.finish(result)
    (output_dir / 'result.json').write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')
    return result


def probe_counts(checkpoint_learning, after_learning, plan):
    """The cumulative probe counts: checkpoint (55) + the single recovery update.

    The checkpoint ``learning`` block is authoritative for the 55 billed updates;
    the resumed runner's block is cumulative 56.  A missing cumulative field, a
    wrong single-update increment or a total that differs from the frozen plan is
    a `ProbeError` (never silently accepted).
    """
    from src.tasks.velocity.v06 import orchestration as orchestration_module
    from src.tasks.velocity.v06 import ppo as ppo_module

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
    from src.tasks.velocity.v06 import spec as spec_module
    from src.tasks.velocity.v06.curriculum import CurriculumManager

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
    """SDS 9.3 probe: 512 train envs + 8 eval envs on L40, 55 updates, save,
    resume, exactly one non-replayed recovery update, plus resource samples.

    The 56 probe updates are billed to the probe ledger only: they are returned
    with ``counts_against_training=False`` and never touch the 30,000 budget.
    """
    import time

    from src.tasks.velocity.v06 import model as model_module
    from src.tasks.velocity.v06 import orchestration as orchestration_module
    from src.tasks.velocity.v06 import ppo as ppo_module
    from src.tasks.velocity.v06 import recovery as recovery_module
    from src.tasks.velocity.v06 import runner as runner_module
    from src.tasks.velocity.v06.curriculum import CurriculumManager

    output_dir = Path(output_dir)
    device = str(plan.get('device') or 'cuda:0')
    seed = int(plan['seed'])
    contract = dict(num_envs=int(plan['num_envs']),
                    eval_num_envs=int(plan['eval_num_envs']),
                    device=device, seed=seed)
    timeline, samples = [], []

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
        model = (model_module.V06ActorCritic.migrate_eth_checkpoint(
            seed=seed, device=device) if mode == 'train'
            else model_module.V06ActorCritic().to(device))
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
        instance = runner_module.V06Runner(
            model=model, ppo=trainer, curriculum=manager, env=env,
            directory=directory, seed=seed, device=device, mode=mode,
            container=container or plan['container'], evaluate=(mode == 'probe'),
            autosave=False, require_resume_evidence=(mode == 'probe'),
            # Both probe instances write the probe container; the source mode stays
            # 'train' so the probe's own formal recovery reader can re-read them.
            source=recovery_module.source_block(seed=seed, device=device,
                                                mode='train'))
        instance.case_runner = build_case_runner(model, contract, phase, None)
        return instance

    from src.tasks.velocity.v06 import spec as spec_module

    runner = build(output_dir / 'train', 'train', phase='R',
                   container=plan['container'],
                   frontier_tile_id=int(spec_module.FINAL_TILE_ID))
    stages = plan['stages']
    for index in range(stages['warmup'] + stages['measured']):
        runner.train_one_update()
        stage = 'warmup' if index < stages['warmup'] else 'measured'
        timeline.append(dict(stage=stage, update=int(runner.update),
                             at=float(time.time())))
        if stage == 'measured':
            sample(runner.update)
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
    return dict(cost=dict(plan['cost']), resources=dict(samples=samples),
                recovery=receipt, phase_timeline=timeline,
                checkpoint=checkpoint['path'],
                checkpoint_sha256=checkpoint['sha256'], counts=counts)


def main(argv=None):
    from src.tasks.velocity.v06.orchestration import main as supervisor_main
    return supervisor_main(argv, worker_factory=build_worker,
                           probe_factory=build_probe_worker)


if __name__ == '__main__':
    raise SystemExit(main())
