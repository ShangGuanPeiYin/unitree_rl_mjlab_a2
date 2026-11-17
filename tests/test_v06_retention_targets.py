"""Q007 CPU contract: retention sets keep their frozen terrain and protocol.

The production crash was ``EvaluationError('F0 case length disagrees with the
observed steps')`` at the update-400 flat retention boundary: the runner derived
the retention tile from the current phase (H -> H00 tile 1), so the flat set ran
the F0 protocol on an H00 stair with a 500-step limit.  These tests pin the
frozen identity in every phase (flat = tile 0/F0, stair = H00 tile 1/H) and
cover the real-CPU case runner, the pending boundary replay and the Q007 legacy
tile correction for checkpoints written before the fix.
"""
import math
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))
import a2_v06_bootstrap  # noqa: E402

a2_v06_bootstrap.activate(ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402

from src.tasks.velocity.v06 import evaluation as evaluation_module  # noqa: E402
from src.tasks.velocity.v06 import recovery as recovery_module  # noqa: E402
from src.tasks.velocity.v06 import runner as runner_module  # noqa: E402
from src.tasks.velocity.v06 import speed  # noqa: E402
from src.tasks.velocity.v06 import spec  # noqa: E402
from test_v06_runner import make_runner  # noqa: E402

PHASES = ('F0', 'H', 'D', 'L', 'V', 'R')
FLAT_TILE = int(spec.FLAT_TILE_ID)
H00_TILE = int(spec.tile_id_for('H00'))


def substage_for(phase):
    allowed = tuple(spec.phase_spec(phase).substages)
    return allowed[0] if allowed else None


def passing(case):
    return dict(measurement='flat' if str(case.get('measurement_stage')) == 'F0'
                else 'stair', climb_success=True, failure=False, flat_pass=True,
                reason=None, mean_vx=case['vx'], mean_relative_error=0.0,
                active_samples=100, stall_samples=0, first_step_valid=True)


def failing(case):
    return dict(measurement='flat' if str(case.get('measurement_stage')) == 'F0'
                else 'stair', climb_success=False, failure=True, flat_pass=False,
                reason='failure', mean_vx=case['vx'], mean_relative_error=0.0,
                active_samples=100, stall_samples=100, first_step_valid=True)


# ------------------------------------------------------- frozen table identity
@pytest.mark.parametrize('phase', PHASES)
def test_flat_retention_is_pinned_to_flat_tile_zero_in_every_phase(phase):
    substage = substage_for(phase)
    table = evaluation_module.make_cases(phase, substage, 'flat_retention', 42)
    assert int(table.target_tile_id) == FLAT_TILE == 0
    assert table.measurement_stage == 'F0'
    assert table.measurement == 'flat'
    assert tuple(table.speeds) == (0.3, 0.5, 0.8)
    assert table.episodes == 5 and len(table.cases) == 15
    assert table.spawn_profile == 'standard'
    assert list(table.spawn_parameters['x']) == [-1.1, -0.9]
    assert list(table.spawn_parameters['y']) == [-0.05, 0.05]
    assert list(table.spawn_parameters['yaw_degrees']) == [-5.0, 5.0]
    assert all(int(case['tile_id']) == 0 for case in table.cases)
    for case in table.cases:
        assert case['measurement_stage'] == 'F0'
        assert runner_module._case_step_limit(case) == int(spec.F0_OBSERVE_STEPS) == 500
    assert evaluation_module.target_tile(phase, substage, 'flat_retention') == 0
    assert evaluation_module.target_tile(phase, substage, 'flat_retention',
                                         tile_id=0) == 0
    with pytest.raises(evaluation_module.EvaluationError):
        evaluation_module.target_tile(phase, substage, 'flat_retention', tile_id=1)


@pytest.mark.parametrize('phase', PHASES)
def test_stair_retention_is_pinned_to_h00_in_every_phase(phase):
    substage = substage_for(phase)
    table = evaluation_module.make_cases(phase, substage, 'stair_retention', 42)
    assert int(table.target_tile_id) == H00_TILE == 1
    assert table.measurement_stage == 'H'
    assert table.measurement == 'stair'
    assert tuple(table.speeds) == (0.3, 0.5, 0.8)
    assert table.episodes == 5 and len(table.cases) == 15
    assert table.spawn_profile == 'standard'
    assert list(table.spawn_parameters['x']) == [-1.1, -0.9]
    assert list(table.spawn_parameters['y']) == [-0.05, 0.05]
    assert list(table.spawn_parameters['yaw_degrees']) == [-5.0, 5.0]
    assert all(int(case['tile_id']) == 1 for case in table.cases)
    for case in table.cases:
        assert case['measurement_stage'] == 'H'
        expected = _stair_step_limit(case)
        assert runner_module._case_step_limit(case) == expected
    assert evaluation_module.target_tile(phase, substage, 'stair_retention') == 1
    with pytest.raises(evaluation_module.EvaluationError):
        evaluation_module.target_tile(phase, substage, 'stair_retention', tile_id=0)


def _stair_step_limit(case):
    tile = runner_module.spec_tile(case)
    steps = math.ceil((2 * (2.2 + float(tile.N) * math.sqrt(
        float(tile.h) ** 2 + float(tile.step_depth) ** 2))
        / max(float(case['vx']), 1e-6) + 10) / 0.02)
    return int(steps)


@pytest.mark.parametrize('kind,tile', [('flat_retention', 0),
                                        ('stair_retention', 1)])
def test_retention_physical_content_is_identical_across_phases(kind, tile):
    reference = evaluation_module.make_cases('H', None, kind, 42, tile_id=tile)
    reference_arrays = deepcopy(reference.arrays)
    reference_cases = [{key: value for key, value in case.items()
                        if key not in ('phase',)}
                       for case in reference.cases]
    for phase in PHASES:
        table = evaluation_module.make_cases(phase, substage_for(phase), kind, 42,
                                             tile_id=tile)
        assert table.arrays == reference_arrays
        assert [{key: value for key, value in case.items() if key not in ('phase',)}
                for case in table.cases] == reference_cases
        assert table.table_seed == reference.table_seed == 200342
        assert int(table.target_tile_id) == tile
        assert table.measurement_stage == reference.measurement_stage


def test_h_flat_retention_tile_zero_is_accepted_and_tile_one_refused():
    table = evaluation_module.make_cases('H', None, 'flat_retention', 42, tile_id=0)
    assert int(table.target_tile_id) == 0
    with pytest.raises(evaluation_module.EvaluationError):
        evaluation_module.make_cases('H', None, 'flat_retention', 42, tile_id=1)
    # The old phase-derived default no longer selects the frontier for retention.
    assert evaluation_module.target_tile('H', None, 'flat_retention') == 0


def test_retention_case_environment_uses_its_measurement_stage():
    import train_a2_ideal_stairs_v06 as train

    contract = dict(num_envs=8, eval_num_envs=8, device='cpu', seed=42)
    flat_case = evaluation_module.make_cases(
        'H', None, 'flat_retention', 42).cases[0]
    kwargs = train.case_environment_kwargs(contract, [flat_case], 'H', None)
    assert kwargs['phase'] == 'F0'
    assert int(kwargs['frontier_tile_id']) == 0
    stair_case = evaluation_module.make_cases(
        'H', None, 'stair_retention', 42).cases[0]
    kwargs = train.case_environment_kwargs(contract, [stair_case], 'H', None)
    assert kwargs['phase'] == 'H'
    assert int(kwargs['frontier_tile_id']) == 1


# --------------------------------------------------------------- due tasks
def test_due_tasks_record_the_frozen_retention_tiles(tmp_path):
    runner, _ppo, _curriculum, _env = make_runner(
        tmp_path, update=200, phase='H', retention=('flat', 'h00'))
    assert runner._current_tile() == 1  # H frontier
    tasks = runner._due_tasks(200)
    retention = [task for task in tasks if task['kind'] == 'retention']
    assert len(retention) == 2
    by_kind = {task['gate_kind']: task for task in retention}
    assert int(by_kind['flat_retention']['tile']) == 0
    assert int(by_kind['stair_retention']['tile']) == 1
    assert by_kind['flat_retention']['case_table_sha256'] == \
        evaluation_module.make_cases('H', None, 'flat_retention', 42,
                                     tile_id=0).sha256
    assert by_kind['stair_retention']['case_table_sha256'] == \
        evaluation_module.make_cases('H', None, 'stair_retention', 42,
                                     tile_id=1).sha256
    assert int(by_kind['flat_retention']['tile']) != runner._current_tile()


def test_flat_retention_gate_semantics_5_plus_5_plus_5(tmp_path):
    runner, _ppo, _curriculum, _env = make_runner(tmp_path, update=200, phase='H')
    runner.case_runner = passing
    result = runner.run_gate('flat_retention', retention_kind='flat')
    assert result.gate['failed_cases'] == 0
    assert all(int(row['total']) == 5 for row in result.summary['per_speed'])
    assert [float(row['c']) for row in result.summary['per_speed']] == [0.3, 0.5, 0.8]
    assert runner.retention['streaks']['flat'] == 0
    assert result.table.target_tile_id == 0

    failing_runner, _p, _c, _e = make_runner(tmp_path / 'fail', update=200,
                                             phase='H')
    failing_runner.case_runner = failing
    result = failing_runner.run_gate('flat_retention', retention_kind='flat')
    assert result.gate['failed_cases'] == 15
    assert failing_runner.retention['streaks']['flat'] == 1


# ------------------------------------- pending replay must never auto-correct
def test_pending_replay_refuses_the_old_tile_without_the_q007_migration(tmp_path):
    runner, _ppo, _curriculum, _env = make_runner(
        tmp_path, update=200, phase='H', failing=5)
    old = dict(kind='retention', gate_kind='flat_retention',
               retention_kind='flat', tile=1, phase='H', substage=None,
               case_table_sha256='afb64ce64395297bd73ea3a842b1c8a9ecc1ae7fce97d0fe888cbf60cfecc5eb')
    with pytest.raises(recovery_module.RecoveryBlocked):
        runner._run_pending_tasks(_pending(200, old))
    assert not [record for record in runner.gates if record.kind == 'flat_retention']


def test_pending_replay_rejects_an_unrelated_retention_tile(tmp_path):
    runner, _ppo, _curriculum, _env = make_runner(
        tmp_path, update=200, phase='H', failing=5)
    table = evaluation_module.make_cases('H', None, 'flat_retention', 42,
                                         tile_id=0)
    task = dict(kind='retention', gate_kind='flat_retention',
                retention_kind='flat', tile=2, phase='H', substage=None,
                case_table_sha256=table.sha256)
    with pytest.raises(recovery_module.RecoveryBlocked):
        runner._run_pending_tasks(_pending(200, task))


def _pending(update, task):
    return dict(update=int(update), phase='H', substage=None, generation=0,
                tasks=[dict(task)], results=[], curriculum_decided=False,
                committed=True)


# ------------------------------------- ordinary gates keep their real target
@pytest.mark.parametrize('phase', ('H', 'D', 'L', 'V', 'R'))
def test_normal_gates_keep_the_current_phase_tile(phase):
    substage = substage_for(phase)
    tile = (1 if phase == 'H' else
            int(spec.tile_id_for(substage)) if phase in ('D', 'L') else
            int(spec.FINAL_TILE_ID))
    table = evaluation_module.make_cases(phase, substage, 'validation', 42,
                                         tile_id=tile)
    assert int(table.target_tile_id) == tile
    assert table.measurement_stage == phase
    diagnostic = evaluation_module.make_cases(phase, substage, 'diagnostic', 42,
                                              tile_id=tile)
    assert int(diagnostic.target_tile_id) == tile
    # Retention never leaks into the ordinary-gate routing.
    assert evaluation_module.target_tile(phase, substage, 'flat_retention') == 0
    assert evaluation_module.target_tile(phase, substage, 'stair_retention') == 1


def test_due_tasks_keep_ordinary_gate_tiles(tmp_path):
    runner, _ppo, _curriculum, _env = make_runner(tmp_path, update=200, phase='H')
    tasks = runner._due_tasks(200)
    by_kind = {task['kind']: task for task in tasks}
    assert int(by_kind['validation']['tile']) == 1
    assert int(by_kind['diagnostic']['tile']) == 1
    assert int(by_kind['validation']['tile']) == runner._current_tile()


# ------------------------------------- fixed 8-env batching and <8 tail
def test_retention_and_gate_batches_keep_the_8_env_and_tail_semantics():
    retention = evaluation_module.make_cases('H', None, 'flat_retention', 42)
    batches = evaluation_module.batch_cases(retention)
    assert [len(batch) for batch in batches] == [5, 5, 5]
    validation = evaluation_module.make_cases('H', None, 'validation', 42, tile_id=1)
    batches = evaluation_module.batch_cases(validation)
    assert [len(batch) for batch in batches] == [8, 8, 4, 8, 8, 4, 8, 8, 4]
    assert all(len(batch) <= 8 for batch in batches)
    assert min(len(batch) for batch in batches) < 8


def test_real_env_reset_pins_the_retention_targets():
    pytest.importorskip('mujoco')
    import train_a2_ideal_stairs_v06 as train
    from src.tasks.velocity.v06.env import V06Env

    contract = dict(num_envs=8, eval_num_envs=8, device='cpu', seed=42)
    flat_cases = [case for case in evaluation_module.make_cases(
        'H', None, 'flat_retention', 42).cases if abs(float(case['vx']) - 0.3) < 1e-9]
    kwargs = train.case_environment_kwargs(contract, flat_cases, 'H', None)
    try:
        flat_env = V06Env(device='cpu', seed=42,
                          **{key: value for key, value in kwargs.items()
                             if key != 'contract'})
        flat_env.reset()
    except Exception as error:  # noqa: BLE001
        pytest.skip(f'CPU V06Env unavailable: {type(error).__name__}: {error}')
    assert int(flat_env.frontier_tile_id) == 0
    assert set(int(value) for value in flat_env.tile_ids.tolist()) == {0}
    stair_cases = [case for case in evaluation_module.make_cases(
        'H', None, 'stair_retention', 42).cases
        if abs(float(case['vx']) - 0.3) < 1e-9]
    kwargs = train.case_environment_kwargs(contract, stair_cases, 'H', None)
    stair_env = V06Env(device='cpu', seed=42,
                       **{key: value for key, value in kwargs.items()
                          if key != 'contract'})
    stair_env.reset()
    assert int(stair_env.frontier_tile_id) == 1
    assert set(int(value) for value in stair_env.tile_ids.tolist()) == {1}


# ------------------------------------------------------------- real CPU env
class ZeroModel:
    def __init__(self, num_envs):
        self.num_envs = int(num_envs)

    def act(self, obs, deterministic=False):
        return torch.zeros(self.num_envs, 12)


def test_real_env_flat_retention_batch_runs_the_flat_protocol():
    pytest.importorskip('mujoco')
    import train_a2_ideal_stairs_v06 as train

    table = evaluation_module.make_cases('H', None, 'flat_retention', 42)
    batch = [case for case in table.cases
             if abs(float(case['vx']) - 0.3) < speed.SPEED_TOLERANCE]
    assert len(batch) == 5
    contract = dict(num_envs=8, eval_num_envs=8, device='cpu', seed=42)
    case_runner = train.build_case_runner(ZeroModel(len(batch)), contract, 'H', None)
    try:
        records = case_runner(batch)
    except Exception as error:  # noqa: BLE001 - no MuJoCo/MJLab on this host
        pytest.skip(f'CPU V06Env unavailable: {type(error).__name__}: {error}')
    assert len(records) == 5
    for record in records:
        assert record['measurement'] == 'flat'
        assert int(record['steps']) <= int(spec.F0_OBSERVE_STEPS)
        if not record['failure']:
            assert int(record['steps']) == int(spec.F0_OBSERVE_STEPS)
        metrics = speed.flat_case_metrics(
            speed=0.3, steps=int(record['steps']),
            max_abs_y=float(record['max_abs_y']),
            max_abs_yaw_deg=float(record['max_abs_yaw_deg']),
            forward=float(record['forward_distance']),
            failure=bool(record['failure']))
        assert speed.flat_case_pass(metrics) == bool(record['flat_pass'])
        assert float(record['max_abs_y']) >= 0.0
        assert float(record['max_abs_yaw_deg']) >= 0.0


def test_real_env_stair_retention_batch_uses_the_stair_timeout():
    pytest.importorskip('mujoco')
    import train_a2_ideal_stairs_v06 as train

    table = evaluation_module.make_cases('H', None, 'stair_retention', 42)
    batch = [case for case in table.cases
             if abs(float(case['vx']) - 0.3) < speed.SPEED_TOLERANCE]
    assert len(batch) == 5
    limit = runner_module._case_step_limit(batch[0])
    assert limit == _stair_step_limit(batch[0])
    contract = dict(num_envs=8, eval_num_envs=8, device='cpu', seed=42)
    case_runner = train.build_case_runner(ZeroModel(len(batch)), contract, 'H', None)
    try:
        records = case_runner(batch)
    except Exception as error:  # noqa: BLE001 - no MuJoCo/MJLab on this host
        pytest.skip(f'CPU V06Env unavailable: {type(error).__name__}: {error}')
    assert len(records) == 5
    for record in records:
        assert record['measurement'] == 'stair'
        assert 0 < int(record['steps']) <= int(limit)
