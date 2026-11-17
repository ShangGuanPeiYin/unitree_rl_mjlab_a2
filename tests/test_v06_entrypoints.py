"""CPU-only v0.6 entry-point contract (SDS 11.1/11.2): the exact flag surface,
recipe-assertion refusal, preflight-only behaviour (no run directory, no CUDA),
``--resume-run`` as the only resume interface and the CPU test launcher."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))
import a2_v06_bootstrap  # noqa: E402

a2_v06_bootstrap.activate(ROOT)

import json  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from src.tasks.velocity.v06 import orchestration as target  # noqa: E402
from src.tasks.velocity.v06 import spec  # noqa: E402

SDS_FLAGS = ('--mode', '--execute', '--recipe', '--seed', '--device', '--num-envs',
             '--eval-num-envs', '--rollout', '--epochs', '--minibatches',
             '--learning-rate', '--gamma', '--entropy-coef', '--initial-std',
             '--std-min', '--std-max', '--initial-vx-min', '--initial-vx-max',
             '--final-vx-min', '--final-vx-max', '--vy', '--wz', '--max-updates',
             '--phase-budgets', '--save-interval', '--diagnostic-interval',
             '--gate-interval', '--init-checkpoint', '--init-sha256',
             '--acceptance-file', '--output-dir')


def sds_argv(output_dir, *, execute=False):
    argv = ['--mode', 'train',
            '--recipe', '0.6-go2refs-r1',
            '--seed', '42',
            '--device', 'cuda:0',
            '--num-envs', '512',
            '--eval-num-envs', '8',
            '--rollout', '48',
            '--epochs', '5',
            '--minibatches', '4',
            '--learning-rate', '0.0003',
            '--gamma', '0.99',
            '--entropy-coef', '0.003',
            '--initial-std', '0.30',
            '--std-min', '0.05',
            '--std-max', '1.00',
            '--initial-vx-min', '0.30',
            '--initial-vx-max', '0.80',
            '--final-vx-min', '0.20',
            '--final-vx-max', '0.60',
            '--vy', '0.0',
            '--wz', '0.0',
            '--max-updates', '30000',
            '--phase-budgets', 'F0=1000,H=12000,D=3000,L=6000,V=4000,R=4000',
            '--save-interval', '100',
            '--diagnostic-interval', '50',
            '--gate-interval', '100',
            '--init-checkpoint', spec.INIT_PATH,
            '--init-sha256', spec.INIT_SHA256,
            '--acceptance-file',
            'logs/a2_ideal_stairs_v06/implementation/V06_IMPLEMENTATION_ACCEPTANCE.md',
            '--output-dir', str(output_dir)]
    if execute:
        argv.insert(2, '--execute')
    return argv


def test_parser_declares_every_sds_flag():
    parser = target.build_parser()
    declared = {action.option_strings[0] for action in parser._actions
                if action.option_strings}
    for flag in SDS_FLAGS:
        assert flag in declared, flag
    assert '--resume-run' in declared
    args = parser.parse_args(sds_argv('/tmp/x'))
    assert args.mode == 'train' and args.execute is False
    assert args.num_envs == 512 and args.learning_rate == 0.0003


def test_preflight_creates_no_run_directory_and_no_cuda(tmp_path):
    output = tmp_path / 'train_seed42_trial01'
    preflight_dir = tmp_path / 'preflight'
    orchestrator = target.V06Orchestrator(
        target.build_parser().parse_args(sds_argv(output)),
        preflight_dir=preflight_dir)
    record = orchestrator.preflight()
    assert record['preflight_only'] is True
    assert record['execute'] is False
    assert record['output_directory_created'] is False
    assert not output.exists()
    assert torch.cuda.is_initialized() is False
    assert record['contract']['mode'] == 'train'
    assert record['contract']['max_updates'] == 30000
    assert record['contract']['phase_budgets'] == spec.PHASE_BUDGETS
    assert record['contract']['initial_vx'] == (0.30, 0.80)
    assert record['contract']['final_vx'] == (0.20, 0.60)
    collapse = record['contract']['diagnostic_collapse_streak']
    assert collapse['value'] == spec.DIAGNOSTIC_COLLAPSE_STREAK == 2
    assert collapse['checks']['reward_config'] is True
    assert collapse['checks']['contract'] is True
    assert len(collapse['reward_config_sha256']) == 64
    path = Path(record['path'])
    assert path.is_file()
    payload = json.loads(path.read_text(encoding='utf-8'))
    assert payload['sds_sha256'] == spec.SDS_SHA256
    assert payload['recipe'] == spec.RECIPE_VERSION
    assert payload['sources']['init_sha256'] == spec.INIT_SHA256


@pytest.mark.parametrize('flag,value', [
    ('--learning-rate', '0.0002'),
    ('--gamma', '0.995'),
    ('--num-envs', '256'),
    ('--rollout', '24'),
    ('--epochs', '4'),
    ('--minibatches', '2'),
    ('--entropy-coef', '0.005'),
    ('--initial-std', '0.15'),
    ('--std-min', '0.10'),
    ('--std-max', '0.30'),
    ('--initial-vx-min', '0.20'),
    ('--final-vx-max', '0.80'),
    ('--vy', '0.1'),
    ('--max-updates', '25000'),
    ('--save-interval', '50'),
    ('--gate-interval', '50'),
    ('--recipe', '0.5-r2'),
    ('--phase-budgets', 'F0=2000,H=12000,D=3000,L=6000,V=4000,R=4000'),
])
def test_recipe_mismatch_is_refused(tmp_path, flag, value):
    argv = sds_argv(tmp_path / 'run')
    index = argv.index(flag)
    argv[index + 1] = value
    orchestrator = target.V06Orchestrator(
        target.build_parser().parse_args(argv),
        preflight_dir=tmp_path / 'preflight')
    with pytest.raises(target.PreflightError):
        orchestrator.preflight()
    assert not (tmp_path / 'run').exists()


def test_wrong_init_checkpoint_is_refused(tmp_path):
    argv = sds_argv(tmp_path / 'run')
    argv[argv.index('--init-sha256') + 1] = 'b' * 64
    orchestrator = target.V06Orchestrator(
        target.build_parser().parse_args(argv),
        preflight_dir=tmp_path / 'preflight')
    with pytest.raises(target.PreflightError):
        orchestrator.preflight()


def test_duplicate_phase_budget_is_refused():
    with pytest.raises(target.PreflightError):
        target.parse_phase_budgets('F0=1000,F0=1000')
    with pytest.raises(target.PreflightError):
        target.parse_phase_budgets('F0')
    assert target.parse_phase_budgets('F0=1000,H=12000') == {'F0': 1000, 'H': 12000}
    assert target.expected_phase_budgets() == spec.PHASE_BUDGETS


def test_resume_run_is_the_only_resume_interface(tmp_path):
    junk = tmp_path / 'checkpoint.pt'
    junk.write_bytes(b'not a checkpoint')
    argv = ['--mode', 'train', '--resume-run', str(junk),
            '--output-dir', str(tmp_path / 'run')]
    orchestrator = target.V06Orchestrator(
        target.build_parser().parse_args(argv),
        preflight_dir=tmp_path / 'preflight')
    with pytest.raises(Exception):
        orchestrator.preflight()
    # A resume may never be combined with a re-initialization flag.
    argv += ['--init-checkpoint', spec.INIT_PATH]
    orchestrator = target.V06Orchestrator(
        target.build_parser().parse_args(argv),
        preflight_dir=tmp_path / 'preflight')
    with pytest.raises(target.PreflightError):
        orchestrator.preflight()


def test_cpu_test_launcher_supports_pattern_and_maxfail():
    import test_a2_ideal_stairs_v06 as launcher

    parser = launcher.parser()
    args = parser.parse_args(['--pattern', 'test_v06_*.py', '--maxfail', '0'])
    assert args.pattern == 'test_v06_*.py'
    assert args.maxfail == 0
    files = launcher._select_files('test_v06_model.py')
    assert [path.name for path in files] == ['test_v06_model.py']
    with pytest.raises(SystemExit):
        launcher._select_files('test_v06_no_such_file.py')


def test_bootstrap_is_idempotent_and_isolated():
    assert a2_v06_bootstrap.is_active(ROOT) is True
    assert a2_v06_bootstrap.active_root() == str(ROOT)
    again = a2_v06_bootstrap.activate(ROOT)
    assert again is sys.modules['src.tasks']
    assert again.__file__ is None


def test_collapse_streak_is_required_without_a_fallback(monkeypatch):
    from src.tasks.velocity.v06 import speed

    assert speed.require_diagnostic_collapse_streak()['value'] == 2
    monkeypatch.delattr(spec, 'DIAGNOSTIC_COLLAPSE_STREAK', raising=True)
    with pytest.raises(speed.MissingConstant):
        speed.diagnostic_collapse_streak()
    with pytest.raises(speed.MissingConstant):
        speed.require_diagnostic_collapse_streak()
    monkeypatch.setattr(spec, 'DIAGNOSTIC_COLLAPSE_STREAK', 1, raising=False)
    with pytest.raises(speed.MissingConstant):
        speed.require_diagnostic_collapse_streak()
    monkeypatch.setattr(spec, 'DIAGNOSTIC_COLLAPSE_STREAK', True, raising=False)
    with pytest.raises(speed.MissingConstant):
        speed.require_diagnostic_collapse_streak()


def test_train_script_uses_the_reliable_connection_when_wandb_is_enabled(tmp_path,
                                                                        monkeypatch):
    import train_a2_ideal_stairs_v06 as train
    from src.tasks.velocity.v06.connection import ReliableConnection
    from src.tasks.velocity.v06.telemetry import V06Telemetry

    monkeypatch.setenv('A2_V06_WANDB', '0')
    offline = train.build_telemetry(tmp_path, 42)
    assert isinstance(offline, V06Telemetry) and not isinstance(offline, ReliableConnection)
    assert offline.directory == tmp_path / 'telemetry'

    class FakeBackend:
        enabled = True
        entity = 'unitree'
        project = 'a2-ideal-stairs-v06'

    online = train.build_telemetry(tmp_path, 42, backend=FakeBackend())
    assert isinstance(online, ReliableConnection)
    assert online.directory == tmp_path / 'telemetry'
    assert online.delivery_timeout == 120
    assert online.max_in_flight == 32

    monkeypatch.setenv('A2_V06_WANDB', '1')
    assert train.wandb_enabled() is True
    monkeypatch.setenv('A2_V06_WANDB', '')
    monkeypatch.delenv('WANDB_API_KEY', raising=False)
    assert train.wandb_enabled() is False


def test_smoke_uses_h00_as_a_tile_not_a_substage():
    import smoke_a2_ideal_stairs_v06 as smoke

    kwargs = smoke.smoke_environment_kwargs()
    assert kwargs['phase'] == 'H'
    assert kwargs['substage'] is None          # H00 is a tile, never a substage
    assert kwargs['level_tick'] == 0
    assert kwargs['tile'] == 'H00'
    assert kwargs['frontier_tile_id'] == spec.tile_id_for('H00') == 1
    terrain = spec.terrain_spec(kwargs['frontier_tile_id'])
    assert terrain.lesson_id == 'H00'
    assert (int(terrain.N), float(terrain.h), float(terrain.step_depth)) == (6, 0.02, 0.30)
    # Phase H has no substages; D25/D20/L12/L20/L40 are the only substages.
    assert tuple(spec.phase_spec('H').substages) == ()
    assert set(spec.phase_spec('D').substages) == {'D25', 'D20'}
    assert set(spec.phase_spec('L').substages) == {'L12', 'L20', 'L40'}

    plan = smoke.smoke_plan()
    assert plan['plan']['phase'] == 'H' and plan['plan']['tile'] == 'H00'
    assert plan['environment']['substage'] is None
    assert plan['environment']['frontier_tile_id'] == 1
    # SDS 9.3 smoke cost is unchanged: 768 control samples / 3072 substeps / 40 Adam.
    assert plan['plan']['control_samples'] == 768
    assert plan['plan']['physics_substeps'] == 3072
    assert plan['plan']['adam_steps'] == 40
    assert plan['plan']['updates'] == 2
    assert plan['resumable'] is False
    assert plan['container'] == 'v06-smoke-state-v1'


def test_smoke_environment_kwargs_never_name_a_terrain_lesson_as_substage():
    import smoke_a2_ideal_stairs_v06 as smoke

    kwargs = smoke.smoke_environment_kwargs()
    for lesson in ('H00', 'H01', 'H15'):
        assert kwargs['substage'] != lesson
    # The env would reject a lesson used as a substage; the smoke never does it.
    assert kwargs['substage'] is None


def test_smoke_history_records_real_per_update_deltas():
    import smoke_a2_ideal_stairs_v06 as smoke

    before = dict(control_samples=0, physics_substeps=0, adam_steps=0, update=0)
    after = dict(control_samples=384, physics_substeps=1536, adam_steps=20, update=1)
    first = smoke.update_cost_deltas(before, after)
    assert first == dict(control_samples=384, physics_substeps=1536,
                         adam_steps=20, update=1)
    final = dict(control_samples=768, physics_substeps=3072, adam_steps=40, update=2)
    second = smoke.update_cost_deltas(after, final)
    assert second == first
    totals = dict(control_samples=first['control_samples'] + second['control_samples'],
                  physics_substeps=(first['physics_substeps']
                                    + second['physics_substeps']),
                  adam_steps=first['adam_steps'] + second['adam_steps'],
                  learning_updates=2)
    plan = smoke.smoke_plan()['plan']
    assert totals['control_samples'] == plan['control_samples'] == 768
    assert totals['physics_substeps'] == plan['physics_substeps'] == 3072
    assert totals['adam_steps'] == plan['adam_steps'] == 40
    with pytest.raises(SystemExit):
        smoke.update_cost_deltas(after, before)  # backwards counter is an error


def test_smoke_history_fields_are_not_episode_counts():
    import inspect
    import smoke_a2_ideal_stairs_v06 as smoke

    source = inspect.getsource(smoke.main)
    assert 'len(records)' not in source
    assert 'smoke_updates' in source
    assert 'cost_totals' in source
    updates_source = inspect.getsource(smoke.smoke_updates)
    assert 'update_cost_deltas' in inspect.getsource(smoke.history_entry)
    assert 'history_entry' in updates_source


def test_smoke_history_builds_through_the_real_worker_path():
    import smoke_a2_ideal_stairs_v06 as smoke

    class FakeTrainer:
        def __init__(self):
            self.counts = dict(control_samples=0, physics_substeps=0,
                               adam_steps=0, update=0)

        @property
        def learning(self):
            return dict(self.counts)

        def train_step(self):
            self.counts['control_samples'] += 384
            self.counts['physics_substeps'] += 1536
            self.counts['adam_steps'] += 20
            self.counts['update'] += 1
            return [], {'loss': 0.5}

    history, learning = smoke.smoke_updates(FakeTrainer(), 2)
    assert [entry['update'] for entry in history] == [1, 2]
    assert [entry['control_samples'] for entry in history] == [384, 384]
    assert [entry['physics_substeps'] for entry in history] == [1536, 1536]
    assert [entry['adam_steps'] for entry in history] == [20, 20]
    assert [entry['total_control_samples'] for entry in history] == [384, 768]
    assert sum(entry['control_samples'] for entry in history) == 768
    assert sum(entry['physics_substeps'] for entry in history) == 3072
    assert sum(entry['adam_steps'] for entry in history) == 40
    assert learning['control_samples'] == 768
    assert learning['physics_substeps'] == 3072
    assert learning['adam_steps'] == 40
    # The merge point itself must not raise the duplicate `update` keyword again.
    entry = smoke.history_entry(3, learning, dict(learning, control_samples=1152,
                                                  physics_substeps=4608,
                                                  adam_steps=60, update=3))
    assert entry['update'] == 3
    assert entry['control_samples'] == 384
