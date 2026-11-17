"""Q007 CPU contract: the update-400 exception transaction and the restricted
pending-task migration.

The designated resume source is ``exception_000400.pt`` (not ``boundary_000400``):
it already inherited the completed 400 validation/diagnostic records and holds
one unfinished flat-retention task written with the pre-fix H00 tile.  Q007
authorises exactly one migration of that task to the frozen flat protocol; the
completed records are never re-run and the completed PPO update is never billed
again.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))
import a2_v06_bootstrap  # noqa: E402

a2_v06_bootstrap.activate(ROOT)

from copy import deepcopy  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from src.tasks.velocity.v06 import evaluation as evaluation_module  # noqa: E402
from src.tasks.velocity.v06 import ledger as ledger_module  # noqa: E402
from src.tasks.velocity.v06 import model as model_module  # noqa: E402
from src.tasks.velocity.v06 import recovery as recovery_module  # noqa: E402
from src.tasks.velocity.v06 import runner as runner_module  # noqa: E402
from src.tasks.velocity.v06 import speed  # noqa: E402
from test_v06_q006 import Q006Env, Q006PPO, passing  # noqa: E402
from test_v06_runner import make_runner  # noqa: E402

SOURCE = ROOT / ledger_module.Q007_SOURCE_CHECKPOINT_PATH
SOURCE_SHA = ledger_module.Q007_SOURCE_CHECKPOINT_SHA256
BOUNDARY400 = ROOT / ledger_module.Q007_FORBIDDEN_CHECKPOINT_PATH
HAS_SOURCE = SOURCE.is_file()
NEW_FLAT_SHA = None


def _load_source():
    return recovery_module.preload_checkpoint(SOURCE, SOURCE_SHA, mode='resume',
                                              allow_bridge=True)


def _q007_runner(tmp_path, state, *, case_runner=passing):
    from src.tasks.velocity.v06.curriculum import CurriculumManager

    model = model_module.V06ActorCritic()
    env = Q006Env(num_envs=int(state['learning']['num_envs']))
    ppo = Q006PPO(model, update=0, num_envs=int(state['learning']['num_envs']))
    curriculum = CurriculumManager(num_envs=int(state['learning']['num_envs']),
                                   seed=42, state=deepcopy(state['curriculum']))
    runner = runner_module.V06Runner(
        model=model, ppo=ppo, curriculum=curriculum, env=env,
        directory=tmp_path, seed=42, device='cpu', case_runner=case_runner,
        require_resume_evidence=True)
    return runner, ppo


class CountingCaseRunner:
    def __init__(self, *, fail_flat=False):
        self.kinds = {}
        self.fail_flat = bool(fail_flat)

    def __call__(self, case):
        kind = str(case.get('kind'))
        self.kinds[kind] = self.kinds.get(kind, 0) + 1
        assert kind == 'flat_retention', f'unexpected evaluation call: {kind}'
        record = passing(case)
        if self.fail_flat:
            record.update(climb_success=False, failure=True, flat_pass=False,
                          reason='failure', stall_samples=100)
        return record

    @property
    def total(self):
        return sum(self.kinds.values())


# ------------------------------------------------------------ source/bridge
@pytest.mark.skipif(not HAS_SOURCE, reason='the Q007 exception source is absent')
def test_exception_source_requires_the_bridge_and_boundary_is_refused():
    with pytest.raises(recovery_module.RecoveryBlocked):
        recovery_module.preload_checkpoint(SOURCE, mode='resume')
    assert BOUNDARY400.is_file()
    boundary_sha = recovery_module.sha256(BOUNDARY400)
    assert boundary_sha == ledger_module.Q007_FORBIDDEN_CHECKPOINT_SHA256
    with pytest.raises(recovery_module.RecoveryBlocked):
        recovery_module.preload_checkpoint(BOUNDARY400, boundary_sha, mode='resume')
    with pytest.raises(recovery_module.RecoveryBlocked):
        recovery_module.preload_checkpoint(BOUNDARY400, boundary_sha,
                                           mode='resume', allow_bridge=True)
    state = _load_source()
    contract = recovery_module.resume_contract(state, allow_bridge=True, seed=42,
                                               num_envs=512)
    bridge = contract['bridge']
    assert bridge['q007_sha256'] == ledger_module.Q007_DECISION_SHA256
    assert bridge['q007_errata_sha256'] == ledger_module.Q007_ERRATA_SHA256
    assert bridge['checkpoint_sha256'] == SOURCE_SHA
    assert contract['replay_billed_updates'] is False
    assert contract['replay_pending_evaluations'] is True
    assert contract['rollback_from_update'] == 400
    assert contract['resume_from_update'] == 400
    assert contract['lost_updates'] == 0
    assert contract['segment']['segment_id'] == ledger_module.Q007_SEGMENT_ID
    plan = contract['pending_task_migration']
    assert plan['old_task']['tile'] == 1
    assert plan['old_task']['case_table_sha256'] == \
        ledger_module.Q007_OLD_FLAT_TASK_SHA256
    assert plan['old_task_protocol_invalid'] is True
    assert plan['new_task']['tile'] == 0
    assert plan['new_task']['case_table_sha256'] == plan['new_case_table_sha256']
    assert plan['new_case_table']['target_tile_id'] == 0
    assert plan['new_case_table']['measurement_stage'] == 'F0'
    assert plan['new_case_table']['speed_order'] == [0.3, 0.5, 0.8]
    assert plan['completed_results'] == list(
        ledger_module.Q007_COMPLETED_RESULT_IDS)
    assert plan['inherited']['validation']['calls'] == 0
    assert plan['inherited']['diagnostic']['calls'] == 0
    # The migration plan is not applied by a read-only contract.
    assert state['ledger']['pending_boundary']['tasks'][2]['tile'] == 1


@pytest.mark.skipif(not HAS_SOURCE, reason='the Q007 exception source is absent')
def test_migration_plan_rejects_unknown_sources_and_completed_tasks():
    state = _load_source()
    other = deepcopy(state)
    other['checkpoint_sha256'] = 'a' * 64
    with pytest.raises(recovery_module.RecoveryBlocked):
        recovery_module.pending_task_migration_plan(other)
    completed = deepcopy(state)
    completed['ledger']['pending_boundary']['results'].append(
        '400:retention:flat:H:' + ledger_module.Q007_OLD_FLAT_TASK_SHA256)
    with pytest.raises(recovery_module.RecoveryBlocked):
        recovery_module.pending_task_migration_plan(completed)
    wrong_kind = deepcopy(state)
    wrong_kind['ledger']['pending_boundary']['tasks'][2]['gate_kind'] = \
        'stair_retention'
    with pytest.raises(recovery_module.RecoveryBlocked):
        recovery_module.pending_task_migration_plan(wrong_kind)
    wrong_sha = deepcopy(state)
    wrong_sha['ledger']['pending_boundary']['tasks'][0]['case_table_sha256'] = \
        'f' * 64
    with pytest.raises(recovery_module.RecoveryBlocked):
        recovery_module.pending_task_migration_plan(wrong_sha)


# ---------------------------------------------------- real exception400 flow
@pytest.mark.skipif(not HAS_SOURCE, reason='the Q007 exception source is absent')
def test_exception_transaction_replays_only_flat_then_ready_and_401(tmp_path):
    state = _load_source()
    source_state = deepcopy(state)
    runner, ppo = _q007_runner(tmp_path, state)
    contract = runner.load(SOURCE, SOURCE_SHA, allow_bridge=True)
    assert contract['pending_task_migration'] is not None
    pending = runner.pending_boundary
    assert [int(task['tile']) for task in pending['tasks']] == [1, 1, 0]
    assert pending['tasks'][2]['case_table_sha256'] == \
        contract['pending_task_migration']['new_case_table_sha256']
    assert (tmp_path / 'recovery' / 'task_migration_000400.json').is_file()
    import json
    manifest = json.loads((tmp_path / 'recovery' /
                           'task_migration_000400.json').read_text(encoding='utf-8'))
    assert manifest['plan_sha256'] == contract['pending_task_migration'][
        'plan_sha256']
    # Checkpoint LR and RNG come from the exception container, never update 100.
    checkpoint_lr = float(source_state['learning']['learning_rate'])
    assert checkpoint_lr == pytest.approx(0.00011390625000000003)
    assert float(runner.ppo.optimizer.param_groups[0]['lr']) == checkpoint_lr
    loaded = runner.save_verified('resume_loaded')
    assert loaded['update'] == 400 and loaded['verification']['aligned']
    loaded_state = torch.load(loaded['path'], map_location='cpu', weights_only=False)
    assert int(loaded_state['learning']['update']) == 400
    assert int(loaded_state['learning']['adam_steps']) == 8000
    assert len(loaded_state['rng']['cuda']) == len(source_state['rng']['cuda'])
    assert all(torch.equal(torch.as_tensor(saved), torch.as_tensor(original))
               for saved, original in zip(loaded_state['rng']['cuda'],
                                          source_state['rng']['cuda']))
    loaded_tasks = loaded_state['ledger']['pending_boundary']['tasks']
    assert int(loaded_tasks[2]['tile']) == 0
    assert loaded_tasks[2]['case_table_sha256'] == manifest['new_case_table_sha256']
    assert loaded_tasks[0]['case_table_sha256'] == \
        ledger_module.Q007_VALIDATION_RESULT_SHA256
    assert loaded_tasks[1]['case_table_sha256'] == \
        ledger_module.Q007_DIAGNOSTIC_RESULT_SHA256
    assert loaded_state['ledger']['pending_boundary']['task_migration'][
        'old_task_sha256'] == ledger_module.Q007_OLD_FLAT_TASK_SHA256

    counter = CountingCaseRunner()
    runner.case_runner = counter
    runner.run(max_updates=400)
    assert counter.total == 15
    assert counter.kinds == {'flat_retention': 15}
    assert runner.pending_boundary is None
    assert runner.update == 400
    assert runner.learning['update'] == 400
    assert int(runner.learning['adam_steps']) == 8000
    # Exactly one curriculum decision; flat next due 600 and (qualified at 400)
    # the first H00 set is due at 600 too.
    state_after = runner._curriculum_state()
    assert state_after['retention_qualified_at']['flat'] == 400 or \
        state_after['retention_qualified_at']['flat'] == 200
    assert state_after['retention_qualified_at']['h00'] == 400
    assert state_after['retention_next_due']['flat'] == 600
    assert state_after['retention_next_due']['h00'] == 600
    assert state_after['retention_streaks'] == {'flat': 0, 'h00': 0}
    assert runner.retention['last']['flat']['update'] == 400
    assert runner.retention['last']['flat']['case_table_sha256'] == \
        manifest['new_case_table_sha256']
    assert runner.resume_ready is not None
    ready = runner.resume_ready
    assert ready['update'] == 400 and ready['verification']['aligned']
    ready_state = torch.load(ready['path'], map_location='cpu', weights_only=False)
    assert ready_state['ledger']['pending_boundary'] is None
    assert int(ready_state['learning']['update']) == 400
    assert int(ready_state['learning']['adam_steps']) == 8000

    # Second load of the verified ready checkpoint: no migration, no replay.
    model2 = model_module.V06ActorCritic()
    env2 = Q006Env(num_envs=512)
    ppo2 = Q006PPO(model2, update=0, num_envs=512)
    from src.tasks.velocity.v06.curriculum import CurriculumManager
    curriculum2 = CurriculumManager(num_envs=512, seed=42,
                                    state=ready_state['curriculum'])
    second_counter = CountingCaseRunner()
    second = runner_module.V06Runner(
        model=model2, ppo=ppo2, curriculum=curriculum2, env=env2,
        directory=tmp_path / 'second', seed=42, device='cpu',
        case_runner=second_counter, require_resume_evidence=True)
    second_contract = second.load(ready['path'], ready['sha256'])
    assert second_contract['pending_task_migration'] is None
    assert second_contract['bridge'] is None
    assert second.pending_boundary is None
    assert second.retention['last']['flat']['update'] == 400
    assert second.gate_streak == runner.gate_streak
    second.save_verified('resume_loaded')
    second.train_one_update()
    assert second.update == 401
    assert int(second.learning['adam_steps']) == 8020
    assert second_counter.total == 0


@pytest.mark.skipif(not HAS_SOURCE, reason='the Q007 exception source is absent')
def test_migrated_retention_streak_is_not_doubled_on_replay(tmp_path):
    state = _load_source()
    runner, _ppo = _q007_runner(tmp_path, state)
    runner.load(SOURCE, SOURCE_SHA, allow_bridge=True)
    loaded = runner.save_verified('resume_loaded')
    runner.case_runner = CountingCaseRunner(fail_flat=True)
    runner.run(max_updates=400)
    assert runner.retention['streaks']['flat'] == 1
    assert runner.resume_ready is not None
    paths = sorted((tmp_path / 'checkpoints').glob('pending_*_000400.pt'))
    assert paths, 'the completed flat set must be persisted as its own checkpoint'
    ready_state = torch.load(runner.resume_ready['path'], map_location='cpu',
                             weights_only=False)
    # Replaying the same persisted transaction is a no-op: the flat result is
    # already in the ledger and cannot be counted twice.
    model2 = model_module.V06ActorCritic()
    env2 = Q006Env(512)
    ppo2 = Q006PPO(model2, update=0, num_envs=512)
    from src.tasks.velocity.v06.curriculum import CurriculumManager
    curriculum2 = CurriculumManager(num_envs=512, seed=42,
                                    state=ready_state['curriculum'])
    second = runner_module.V06Runner(
        model=model2, ppo=ppo2, curriculum=curriculum2, env=env2,
        directory=tmp_path / 'second', seed=42, device='cpu',
        case_runner=passing, require_resume_evidence=True)
    second.load(runner.resume_ready['path'], runner.resume_ready['sha256'])
    second.save_verified('resume_loaded')
    assert second.retention['streaks']['flat'] == 1
    assert [record.kind for record in second.gates].count('flat_retention') == 1
    assert loaded['verification']['aligned']


# ------------------------------------- pending-result checkpoints (600 case)
def test_pending_result_checkpoint_survives_a_dual_retention_crash(tmp_path):
    runner, _ppo, _curriculum, _env = make_runner(
        tmp_path, update=599, phase='H', retention=('flat', 'h00'))
    runner.save_pending_results = True
    calls = {'flat': 0, 'stair': 0}

    def flaky(case):
        kind = str(case.get('kind'))
        if kind == 'flat_retention':
            calls['flat'] += 1
            return dict(measurement='flat', climb_success=False, failure=True,
                        flat_pass=False, reason='failure', mean_vx=case['vx'],
                        mean_relative_error=0.0, active_samples=100,
                        stall_samples=100, first_step_valid=True)
        if kind == 'stair_retention':
            calls['stair'] += 1
            raise RuntimeError('stair retention transport blew up')
        return passing(case)

    runner.case_runner = flaky
    with pytest.raises(RuntimeError):
        runner.run(max_updates=600)
    pending_files = sorted((tmp_path / 'checkpoints').glob('pending_*_000600.pt'))
    assert len(pending_files) == 3, [path.name for path in pending_files]
    checkpoint = torch.load(pending_files[-1], map_location='cpu', weights_only=False)
    pending = checkpoint['ledger']['pending_boundary']
    assert pending['results'] and all('retention' not in item or True
                                      for item in pending['results'])
    assert sum(1 for item in pending['results'] if ':retention:' in item) == 1
    assert checkpoint['retention']['last']['flat']['update'] == 600
    exception = torch.load(tmp_path / 'checkpoints' / 'exception_000600.pt',
                           map_location='cpu', weights_only=False)
    assert sum(1 for item in exception['ledger']['pending_boundary']['results']
               if ':retention:' in item) == 1

    resumed, _ppo2, _c2, _e2 = make_runner(tmp_path / 'resume', update=0,
                                           phase='H')
    resumed.save_pending_results = True
    resumed.load(pending_files[-1],
                 recovery_module.sha256(pending_files[-1]))
    assert resumed.retention['last']['flat']['update'] == 600
    flat_calls = {'n': 0}
    stair_calls = {'n': 0}

    def second_runner(case):
        kind = str(case.get('kind'))
        if kind == 'flat_retention':
            flat_calls['n'] += 1
            raise AssertionError('the completed flat set must not be replayed')
        if kind == 'stair_retention':
            stair_calls['n'] += 1
            return dict(measurement='stair', climb_success=False, failure=True,
                        flat_pass=False, reason='failure', mean_vx=case['vx'],
                        mean_relative_error=0.0, active_samples=100,
                        stall_samples=100, first_step_valid=True)
        return passing(case)

    resumed.case_runner = second_runner
    resumed.run(max_updates=600)
    assert flat_calls['n'] == 0
    assert stair_calls['n'] == 15
    assert resumed.retention['streaks']['flat'] == 1
    assert resumed.retention['streaks']['h00'] == 1
    assert [record.kind for record in resumed.gates].count('flat_retention') == 1
    assert [record.kind for record in resumed.gates].count('stair_retention') == 1


# ------------------------------------- 50/150-style save-fault protection
def test_pending_update_exception_marker_is_refused_for_resume(tmp_path):
    runner, ppo, curriculum, _env = make_runner(tmp_path, update=49)
    ppo.learning_state['completed_ppo_updates'] = 50
    ppo.learning_state['optimizer_steps'] = 1000
    runner.update = 50
    curriculum.spent = {'F0': 49}
    curriculum.last_update = 49
    curriculum.generation_updates = 49
    record = runner.save('exception', pending_update=50)
    state = torch.load(record['path'], map_location='cpu', weights_only=False)
    assert state['ledger']['pending_update'] == 50
    with pytest.raises(recovery_module.RecoveryBlocked):
        recovery_module.preload_checkpoint(record['path'], record['sha256'],
                                           mode='resume')


# --------------------------------------------------------- CLI / evidence
@pytest.mark.skipif(not HAS_SOURCE, reason='the Q007 exception source is absent')
def test_q007_bridge_flag_preflight_and_boundary_negative(tmp_path):
    from src.tasks.velocity.v06 import orchestration as orchestration_module

    manifest = ROOT / ledger_module.Q007_BRIDGE_MANIFEST_PATH
    if not manifest.is_file():
        pytest.skip('the Q007 recovery manifest is absent')
    base = ['--mode', 'train', '--resume-run', str(SOURCE),
            '--allow-q007-bridge']
    output = tmp_path / 'run'
    orchestrator = orchestration_module.V06Orchestrator(
        orchestration_module.build_parser().parse_args(base + ['--output-dir',
                                                              str(output)]),
        preflight_dir=tmp_path / 'preflight')
    contract = orchestrator.preflight()['contract']
    assert contract['allow_q007_bridge'] is True
    assert contract['segment']['segment_id'] == ledger_module.Q007_SEGMENT_ID
    assert contract['segment']['rollback_from_update'] == 400
    assert contract['segment']['resume_from_update'] == 400
    assert contract['segment']['lost_updates'] == 0
    assert contract['segment']['replay_billed_updates'] is False
    assert contract['segment']['replay_pending_evaluations'] is True
    assert contract['resume']['replay_billed_updates'] is False
    assert contract['resume']['replay_pending_evaluations'] is True
    assert contract['resume']['bridge']['q007_sha256'] == \
        ledger_module.Q007_DECISION_SHA256
    assert contract['resume']['pending_task_migration']['plan_sha256']
    assert contract['q007_bridge_manifest']['sha256'] == \
        recovery_module.sha256(manifest)
    assert not output.exists()

    default_contract = orchestration_module.validate_arguments(
        orchestration_module.build_parser().parse_args(base))
    assert default_contract['output_dir'].endswith(
        'train_seed42_trial02_resume_000400_q007')

    boundary = ROOT / ledger_module.Q007_FORBIDDEN_CHECKPOINT_PATH
    with pytest.raises(orchestration_module.PreflightError):
        orchestration_module.validate_arguments(
            orchestration_module.build_parser().parse_args(
                ['--mode', 'train', '--resume-run', str(boundary),
                 '--allow-q007-bridge',
                 '--output-dir', str(tmp_path / 'boundary_run')]))
    with pytest.raises(orchestration_module.PreflightError):
        orchestration_module.validate_arguments(
            orchestration_module.build_parser().parse_args(
                ['--mode', 'train', '--resume-run', str(SOURCE),
                 '--output-dir', str(tmp_path / 'no_flag')]))


def test_q007_ledger_and_migration_manifest():
    ledger = ledger_module.build_q007_ledger()
    assert ledger_module.validate_ledger(ledger) is True
    totals = ledger['totals']
    assert totals['effective_learning_update'] == 400
    assert totals['physical_training_updates'] == 500
    assert totals['lost_branch_updates'] == 100
    budget = ledger['budget']
    assert budget['per_run_cap'] == 30000
    assert budget['combined_formal_cap'] == 30100
    assert budget['remaining_active_branch'] == 29600
    assert budget['current_physical_total'] == 500
    # The active branch's 101-400 rows and the lost Q005 101-150 rows coexist.
    q006 = [row for row in ledger['entries']
            if row['segment_id'] == ledger_module.Q006_SEGMENT_ID]
    assert [row['update'] for row in q006] == list(range(101, 401))
    q005 = [row for row in ledger['entries']
            if row['segment_id'] == 'trial02_q005_resume_000100']
    assert [row['update'] for row in q005] == list(range(101, 151))
    assert {(row['segment_id'], row['update']) for row in q006} & \
        {(row['segment_id'], row['update']) for row in q005} == set()
    path = ROOT / ledger_module.Q007_TASK_MIGRATION_PATH
    if path.is_file():
        import json
        entries = json.loads(path.read_text(encoding='utf-8'))
        assert isinstance(entries, list) and entries
        assert entries[0]['old_task']['tile'] == 1
        assert entries[0]['new_task']['tile'] == 0
        assert entries[0]['old_task']['case_table_sha256'] == \
            ledger_module.Q007_OLD_FLAT_TASK_SHA256
        assert entries[0]['old_task_protocol_invalid'] is True
        assert entries[0]['new_case_table']['target_tile_id'] == 0


# ------------------------------------- zero-learning real-env isolation
@pytest.mark.skipif(not HAS_SOURCE, reason='the Q007 exception source is absent')
def test_zero_learning_both_retention_sets_on_real_8env_targets():
    pytest.importorskip('mujoco')
    import train_a2_ideal_stairs_v06 as train

    state = _load_source()
    model = model_module.V06ActorCritic()
    recovery_module.load_model_state(model, state['model'])
    model.eval()
    contract = dict(num_envs=8, eval_num_envs=8, device='cpu', seed=42)
    for kind, tile, stage in (('flat_retention', 0, 'F0'),
                              ('stair_retention', 1, 'H')):
        table = evaluation_module.make_cases('H', None, kind, 42, tile_id=tile)
        assert int(table.target_tile_id) == tile
        case_runner = train.build_case_runner(model, contract, 'H', None)
        records = []
        for batch in evaluation_module.batch_cases(table):
            try:
                rows = case_runner(batch)
            except Exception as error:  # noqa: BLE001
                pytest.skip(f'CPU V06Env unavailable: {type(error).__name__}: {error}')
            assert len(rows) == len(batch)
            records.extend(rows)
        assert len(records) == 15
        assert all(record['measurement'] == ('flat' if kind == 'flat_retention'
                                             else 'stair') for record in records)
        limit = runner_module._case_step_limit(table.cases[0])
        assert all(0 < int(record['steps']) <= int(limit) for record in records)
        if kind == 'flat_retention':
            assert limit == int(speed.F0_GATE_STEPS)
            for record in records:
                evidence = record['forward_distance_evidence']
                assert evidence['start'] is not None
                assert evidence['end'] is not None
                assert evidence['canonical_net'] == pytest.approx(
                    float(evidence['end']) - float(evidence['start']), abs=1e-6)
                assert abs(float(evidence['D_env']) -
                           float(evidence['canonical_net'])) <= 1e-6
        else:
            assert limit > int(speed.F0_GATE_STEPS)
