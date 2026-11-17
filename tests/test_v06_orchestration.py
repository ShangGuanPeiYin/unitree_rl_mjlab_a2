"""CPU-only v0.6 orchestration contract (SDS 8.5/11.2): preflight side effects,
directory refusal, the four resource limits, the single GPU worker lock, the
separated user-stop state and the telemetry panel surface."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))
import a2_v06_bootstrap  # noqa: E402

a2_v06_bootstrap.activate(ROOT)

import json  # noqa: E402
from functools import partial  # noqa: E402

import torch  # noqa: E402

import pytest  # noqa: E402

from src.tasks.velocity.v06 import orchestration as target  # noqa: E402
from src.tasks.velocity.v06 import recovery  # noqa: E402
from src.tasks.velocity.v06 import spec  # noqa: E402
from src.tasks.velocity.v06 import telemetry  # noqa: E402
from test_v06_entrypoints import sds_argv  # noqa: E402
from test_v06_schema import make_state  # noqa: E402


_real_validate = target.validate_arguments


def good_snapshot(**overrides):
    snapshot = dict(timestamp=0.0, device=0, whole_device_mib=1000, process_mib=1000,
                    host_available_mib=8000, disk_free_gib=50.0, gpu_query=[])
    snapshot.update(overrides)
    return snapshot


def test_resource_limits_match_the_frozen_spec():
    limits, source = target.resource_limits()
    assert source == 'spec.RESOURCE_LIMITS'
    assert limits['whole_device_mib'] == 7168
    assert limits['process_mib'] == 5120
    assert limits['host_available_mib'] == 4096
    assert limits['disk_free_gib'] == 10
    assert spec.RESOURCE_LIMITS['single_gpu_worker'] is True


@pytest.mark.parametrize('field,value,blocking', [
    ('whole_device_mib', 7169, 'whole_device_mib'),
    ('process_mib', 5121, 'process_mib'),
    ('host_available_mib', 4095, 'host_available_mib'),
    ('disk_free_gib', 9.9, 'disk_free_gib'),
])
def test_each_resource_limit_is_enforced(field, value, blocking):
    result = target.check_resources(good_snapshot(**{field: value}))
    assert result['ok'] is False
    assert result['blocking'] == [blocking]
    assert result['checks'][field]['ok'] is False


def test_resources_pass_at_the_exact_limits():
    result = target.check_resources(good_snapshot(whole_device_mib=7168,
                                                  process_mib=5120,
                                                  host_available_mib=4096,
                                                  disk_free_gib=10.0))
    assert result['ok'] is True and result['blocking'] == []


def execute_args(tmp_path):
    return target.build_parser().parse_args(sds_argv(tmp_path / 'run', execute=True))


def test_execute_creates_the_run_and_runs_one_worker(tmp_path, monkeypatch):
    calls = []

    def worker(contract, output):
        calls.append((contract, output))
        return {'state': 'budget_exhausted'}

    orchestrator = target.V06Orchestrator(
        execute_args(tmp_path), preflight_dir=tmp_path / 'preflight',
        resource_probe=lambda directory=None: good_snapshot(),
        worker_factory=worker)
    # The acceptance report is written by the independent reviewer before a run;
    # this CPU test bypasses only that on-disk precondition.
    monkeypatch.setattr(target, 'validate_arguments',
                        partial(_real_validate, require_acceptance=False))
    record = orchestrator.preflight()
    assert record['execute'] is True
    result = orchestrator.execute()
    assert len(calls) == 1
    output = Path(result['contract']['output_dir'])
    assert output.is_dir()
    assert (output / 'control.json').is_file()
    assert (output / 'contract.json').is_file()
    # A terminal worker state is propagated verbatim (never a fake 'finished').
    assert result['status'] in ('finished', 'stopped', 'budget_exhausted')
    assert result['resources']['ok'] is True
    assert not (tmp_path / 'preflight' / 'gpu_worker.json').exists()


def test_execute_refuses_an_existing_output_directory(tmp_path, monkeypatch):
    output = tmp_path / 'run'
    output.mkdir()
    orchestrator = target.V06Orchestrator(
        execute_args(tmp_path), preflight_dir=tmp_path / 'preflight',
        resource_probe=lambda directory=None: good_snapshot(),
        worker_factory=lambda contract, directory: {})
    monkeypatch.setattr(target, 'validate_arguments',
                        partial(_real_validate, require_acceptance=False))
    with pytest.raises(target.AdmissionError):
        orchestrator.preflight()
    with pytest.raises(target.AdmissionError):
        orchestrator.execute()


def test_execute_refuses_insufficient_resources(tmp_path, monkeypatch):
    orchestrator = target.V06Orchestrator(
        execute_args(tmp_path), preflight_dir=tmp_path / 'preflight',
        resource_probe=lambda directory=None: good_snapshot(whole_device_mib=9000),
        worker_factory=lambda contract, directory: {})
    monkeypatch.setattr(target, 'validate_arguments',
                        partial(_real_validate, require_acceptance=False))
    orchestrator.preflight()
    with pytest.raises(target.AdmissionError):
        orchestrator.execute()
    assert not (tmp_path / 'run').exists()


def test_single_gpu_worker_lock(tmp_path):
    registry = tmp_path / 'gpu_worker.json'
    first = target.acquire_single_worker('cuda:0', path=registry)
    assert first.is_file()
    with pytest.raises(target.AdmissionError):
        target.acquire_single_worker('cuda:1', path=registry, pid=999999)
    # The recorded pid is alive only if it is this process; a stale pid is reusable.
    target.release_single_worker(registry)
    assert not registry.exists()
    target.acquire_single_worker('cuda:0', path=registry, pid=999999)
    target.acquire_single_worker('cuda:0', path=registry, pid=123456)
    target.release_single_worker(registry, pid=123456)
    assert not registry.exists()


def test_user_stop_is_not_a_training_or_communication_failure(tmp_path, monkeypatch):
    orchestrator = target.V06Orchestrator(
        execute_args(tmp_path), preflight_dir=tmp_path / 'preflight',
        resource_probe=lambda directory=None: good_snapshot(),
        worker_factory=lambda contract, directory: {})
    monkeypatch.setattr(target, 'validate_arguments',
                        partial(_real_validate, require_acceptance=False))
    orchestrator.preflight()
    output = tmp_path / 'run'
    output.mkdir()
    record = orchestrator.request_stop('user_requested_stop', output=output)
    assert record['action'] == 'stop'
    assert record['stop_state'] == 'user_requested_stop'
    assert record['failure_class'] == 'user'
    assert record['resume_authorized'] is False
    on_disk = json.loads((output / 'control.json').read_text(encoding='utf-8'))
    assert on_disk['stop_state'] == 'user_requested_stop'


def test_preflight_only_entrypoint_returns_zero(tmp_path):
    code = target.main(sds_argv(tmp_path / 'run'))
    assert code == 0
    assert not (tmp_path / 'run').exists()


def test_resume_preflight_reads_the_checkpoint_contract(tmp_path):
    state = make_state(update=5)
    path = tmp_path / 'update_000005.pt'
    sha = recovery.atomic_save(path, state)
    argv = ['--mode', 'train', '--resume-run', str(path),
            '--output-dir', str(tmp_path / 'run')]
    orchestrator = target.V06Orchestrator(
        target.build_parser().parse_args(argv),
        preflight_dir=tmp_path / 'preflight')
    record = orchestrator.preflight()
    assert record['contract']['mode'] == 'resume'
    assert record['resume']['update'] == 5
    assert record['resume']['new_episode'] is True
    assert record['resume']['replay_billed_updates'] is False
    assert record['contract']['resume_run'] == str(path)
    assert sha


def test_telemetry_panels_are_complete_and_never_feed_back(tmp_path):
    telemetry_instance = telemetry.V06Telemetry(tmp_path / 'telemetry')
    panels = telemetry_instance.panels()
    assert panels['panels'] == list(telemetry.PANELS)
    for name in ('train', 'reward', 'ppo_grad', 'curriculum', 'gate', 'retention',
                 'resource'):
        assert name in panels['panels']
    assert panels['ppo_gradient_groups'] == ['actor', 'critic', 'fusion', 'std']
    assert 'tracking_lin_vel' in panels['reward_terms']
    telemetry_instance.start('run', {'seed': 42})
    record = telemetry_instance.update(
        dict(loss=1.0, kl=0.01, lr=3e-4,
             grad_norm_actor={'min': 1.0, 'mean': 2.0, 'max': 3.0},
             reward_terms={'tracking_lin_vel': 0.5}),
        dict(phase='H', substage=None, actions=['gate']))
    assert record['train']['loss'] == 1.0
    flat = telemetry.flatten_numeric({'a': {'b': 1.0}, 'c': 2.0})
    assert flat == {'a/b': 1.0, 'c': 2.0}
    with pytest.raises(telemetry.TelemetryError):
        telemetry.finite_metrics({'loss': float('nan')})
    assert (tmp_path / 'telemetry' / 'telemetry.jsonl').is_file()
    summary = telemetry_instance.finish({'state': 'complete', 'update': 10,
                                         'failure_class': None})
    assert summary['state'] == 'complete'


# ------------------------------------------------- Q002-4 state source
def test_stop_reasons_come_from_the_frozen_spec():
    assert tuple(target.spec_module().STOP_STATES) == (
        'behavior_regression', 'behavior_collapse', 'curriculum_stalled',
        'budget_exhausted', 'gate_failed', 'user_requested_stop',
        'monitoring_blocked', 'resource_insufficient')
    assert tuple(target.spec_module().LIFECYCLE_STATES) == (
        'running', 'ready_for_final_test', 'complete')
    orchestrator = target.V06Orchestrator(
        target.build_parser().parse_args(['--mode', 'train',
                                          '--output-dir', '/tmp/x']))
    with pytest.raises(target.AdmissionError):
        orchestrator.request_stop('pilot_complete', output=Path('/tmp'))
    record = orchestrator.request_stop('monitoring_blocked', output=Path('/tmp'))
    assert record['stop_state'] == 'monitoring_blocked'
    assert record['failure_class'] == 'communication'


def test_worker_engineering_error_is_typed_and_never_complete(tmp_path, monkeypatch):
    def worker(contract, output):
        raise RuntimeError('simulator unavailable')

    orchestrator = target.V06Orchestrator(
        execute_args(tmp_path), preflight_dir=tmp_path / 'preflight',
        resource_probe=lambda directory=None: good_snapshot(), worker_factory=worker)
    monkeypatch.setattr(target, 'validate_arguments',
                        partial(_real_validate, require_acceptance=False))
    orchestrator.preflight()
    with pytest.raises(RuntimeError):
        orchestrator.execute()
    output = tmp_path / 'run'
    control = json.loads((output / 'control.json').read_text(encoding='utf-8'))
    error = json.loads((output / 'error.json').read_text(encoding='utf-8'))
    assert control['action'] == 'failed'
    assert control['error_type'] == 'RuntimeError'
    assert control['stop_state'] is None
    assert error['error_type'] == 'RuntimeError'
    assert control.get('stop_state') != 'complete'


def test_monitoring_failure_is_a_monitoring_blocked_stop(tmp_path, monkeypatch):
    class MonitoringBlocked(RuntimeError):
        pass

    def worker(contract, output):
        raise MonitoringBlocked('remote delivery unconfirmed')

    orchestrator = target.V06Orchestrator(
        execute_args(tmp_path), preflight_dir=tmp_path / 'preflight',
        resource_probe=lambda directory=None: good_snapshot(), worker_factory=worker)
    monkeypatch.setattr(target, 'validate_arguments',
                        partial(_real_validate, require_acceptance=False))
    orchestrator.preflight()
    with pytest.raises(MonitoringBlocked):
        orchestrator.execute()
    control = json.loads((tmp_path / 'run' / 'control.json').read_text(
        encoding='utf-8'))
    assert control['stop_state'] == 'monitoring_blocked'
    assert control['failure_class'] == 'communication'
    assert control['error_type'] == 'MonitoringBlocked'


# ======================================================= ROOT regression
def test_root_is_the_project_root_not_src():
    root = Path(__file__).resolve().parents[1]
    assert target.ROOT == root
    assert (target.ROOT / 'doc/sds/SDS_A2_IdealHeightMap_Stairs40_v0.6.md').is_file()
    assert not str(target.ROOT).endswith('/src')


def test_preflight_writes_under_the_project_root_not_src(tmp_path):
    root = tmp_path / 'root'
    root.mkdir()
    argv = ['--mode', 'train', '--seed', '42', '--device', 'cuda:0',
            '--output-dir', 'logs/a2_ideal_stairs_v06/train_seed42_trial01',
            '--init-checkpoint', spec.INIT_PATH,
            '--init-sha256', spec.INIT_SHA256]
    orchestrator = target.V06Orchestrator(
        target.build_parser().parse_args(argv), root=root)
    record = orchestrator.preflight()
    preflight = Path(record['path'])
    assert preflight.is_file()
    assert preflight.resolve().is_relative_to((root / 'logs').resolve())
    assert 'src/logs' not in str(preflight)
    assert not (root / 'src').exists()
    assert record['contract']['output_dir'] == str(
        root / 'logs/a2_ideal_stairs_v06/train_seed42_trial01')
    assert not (root / 'logs/a2_ideal_stairs_v06/train_seed42_trial01').exists()


def test_existing_output_directory_is_refused_and_untouched(tmp_path, monkeypatch):
    output = tmp_path / 'run'
    output.mkdir()
    sentinel = output / 'keep.txt'
    sentinel.write_text('do not touch', encoding='utf-8')
    orchestrator = target.V06Orchestrator(
        execute_args(tmp_path), preflight_dir=tmp_path / 'preflight',
        resource_probe=lambda directory=None: good_snapshot(),
        worker_factory=lambda contract, directory: {})
    monkeypatch.setattr(target, 'validate_arguments',
                        partial(_real_validate, require_acceptance=False))
    with pytest.raises(target.AdmissionError):
        orchestrator.execute()
    assert sentinel.read_text(encoding='utf-8') == 'do not touch'
    assert sorted(path.name for path in output.iterdir()) == ['keep.txt']


# ============================================================ probe (SDS 9.3)
def probe_receipt(plan):
    counts = dict(plan['cost'], counts_against_training=False,
                  training_budget_untouched=int(plan['training_budget_untouched']))
    return dict(resume_mode='new_episode', resumed_from_update=55,
                updates_after=1, billed_updates_not_replayed=True,
                initialized_from_eth_again=False,
                container=target.probe_container(),
                checkpoint_sha256='a' * 64,
                curriculum_state={'phase': 'R'}, counts=counts)


def probe_samples(count=50, **overrides):
    sample = dict(whole_device_mib=6000, process_mib=4000,
                  process_mib_method='compute_apps',
                  host_available_mib=8000, disk_free_gib=50.0)
    sample.update(overrides)
    return [dict(update=index + 6, **sample) for index in range(count)]


def probe_root(tmp_path):
    root = tmp_path / 'root'
    root.mkdir(parents=True, exist_ok=True)
    return root


def probe_output(tmp_path):
    return probe_root(tmp_path) / 'logs/a2_ideal_stairs_v06/implementation/probe/probe_seed42'


def probe_args(tmp_path):
    return target.build_parser().parse_args(
        ['--mode', 'probe', '--execute', '--seed', '42', '--device', 'cuda:0',
         '--output-dir', str(probe_output(tmp_path))])


def test_probe_plan_cost_identity_and_budget_isolation():
    plan = target.probe_plan()
    assert plan['updates'] == 56
    assert plan['stages'] == dict(warmup=5, measured=50, recovery=1)
    assert plan['cost']['learning_updates'] == 56
    assert plan['cost']['control_samples'] == 512 * 48 * 56 == 1_376_256
    assert plan['cost']['physics_substeps'] == 4 * 1_376_256
    assert plan['cost']['adam_updates'] == 20 * 56
    assert plan['num_envs'] == 512 and plan['eval_num_envs'] == 8
    assert plan['tile'] == 'L40' and plan['tile_id'] == spec.FINAL_TILE_ID
    assert plan['counts_against_training'] is False
    assert plan['training_budget_untouched'] == 30000
    ledger = target.probe_ledger(plan)
    assert ledger['counts_against_training'] is False
    assert ledger['training_budget_untouched'] == 30000
    assert ledger['training_budget_consumed'] == 0
    assert ledger['resumable'] is False
    assert ledger['container'] == 'v06-probe-state-v1'


def test_probe_without_execute_has_no_side_effects(tmp_path):
    root = probe_root(tmp_path)
    orchestrator = target.V06Orchestrator(probe_args(tmp_path), root=root,
                                          preflight_dir=tmp_path / 'preflight')
    record = orchestrator.preflight()
    assert record['mode'] == 'probe'
    assert record['execute'] is True
    assert record['probe']['updates'] == 56
    assert record['probe']['cost']['control_samples'] == 1_376_256
    assert not probe_output(tmp_path).exists()
    assert not (root / 'src').exists()
    assert record['output_directory_created'] is False
    path = Path(record['path'])
    assert path.name == 'preflight_probe_seed42.json'
    assert 'preflight' in str(path)


def test_probe_execute_writes_engineering_evidence(tmp_path, monkeypatch):
    calls = []

    def factory(plan, output):
        calls.append((plan, output))
        return dict(cost=dict(plan['cost']),
                    resources=dict(samples=probe_samples()),
                    recovery=probe_receipt(plan),
                    phase_timeline=[dict(stage='measured', update=55),
                                    dict(stage='recovery', update=56)],
                    checkpoint=str(Path(output) / 'checkpoints/probe_000055.pt'),
                    checkpoint_sha256='b' * 64)

    orchestrator = target.V06Orchestrator(
        probe_args(tmp_path), root=probe_root(tmp_path),
        preflight_dir=tmp_path / 'preflight',
        resource_probe=lambda directory=None: good_snapshot(),
        probe_factory=factory)
    orchestrator.preflight()
    result = orchestrator.execute()
    assert len(calls) == 1
    assert result['status'] == 'engineering_check_only'
    probe = result['probe']
    output = Path(result['contract']['output_dir'])
    assert (output / 'probe.json').is_file()
    assert (output / 'preflight.json').is_file()
    assert probe['cost']['control_samples'] == 1_376_256
    assert probe['cost']['learning_updates'] == 56
    assert probe['recovery']['resume_mode'] == 'new_episode'
    assert probe['recovery']['resumed_from_update'] == 55
    assert probe['recovery']['updates_after'] == 1
    assert probe['recovery']['billed_updates_not_replayed'] is True
    assert probe['counts_against_training'] is False
    assert probe['training_budget_untouched'] == 30000
    assert probe['acceptance_scope'] == 'engineering_check_only'
    assert probe['status'] == 'engineering_check_only'
    assert probe['process_mib_method'] == 'compute_apps'
    assert probe['resources']['measured']['metrics']['process_mib']['method'] == \
        'compute_apps'
    assert probe['resumable'] is False
    assert probe['container'] == 'v06-probe-state-v1'
    assert len(probe['probe_sha256']) == 64
    metrics = probe['resources']['measured']['metrics']
    for name in ('whole_device_mib', 'process_mib', 'host_available_mib',
                 'disk_free_gib'):
        assert metrics[name]['median'] is not None
        assert metrics[name]['ok'] is True
        assert metrics[name]['min'] <= metrics[name]['median'] <= metrics[name]['max']
    # The probe writes nothing to the training budget or the training run dir.
    assert not (output.parent.parent / 'train_seed42_trial01').exists()
    with pytest.raises(target.AdmissionError):
        target.V06Orchestrator(
            probe_args(tmp_path), root=probe_root(tmp_path),
            preflight_dir=tmp_path / 'preflight2',
            resource_probe=lambda directory=None: good_snapshot(),
            probe_factory=factory).execute()


def test_probe_resource_insufficient_path(tmp_path):
    def factory(plan, output):
        return dict(cost=dict(plan['cost']),
                    resources=dict(samples=probe_samples(whole_device_mib=8000)),
                    recovery=probe_receipt(plan), phase_timeline=[])

    orchestrator = target.V06Orchestrator(
        probe_args(tmp_path), root=probe_root(tmp_path),
        preflight_dir=tmp_path / 'preflight',
        resource_probe=lambda directory=None: good_snapshot(),
        probe_factory=factory)
    orchestrator.preflight()
    result = orchestrator.execute()
    assert result['status'] == 'resource_insufficient'
    assert result['probe']['status'] == 'resource_insufficient'
    assert 'whole_device_mib' in result['probe']['resources']['blocking']


def test_probe_recovery_validation_rejects_replays_and_wrong_containers():
    plan = target.probe_plan()
    assert target.validate_probe_recovery(probe_receipt(plan), plan) is True
    receipt = probe_receipt(plan)
    receipt['billed_updates_not_replayed'] = False
    with pytest.raises(target.ProbeError):
        target.validate_probe_recovery(receipt, plan)
    receipt = probe_receipt(plan)
    receipt['updates_after'] = 2
    with pytest.raises(target.ProbeError):
        target.validate_probe_recovery(receipt, plan)
    receipt = probe_receipt(plan)
    receipt['resumed_from_update'] = 54
    with pytest.raises(target.ProbeError):
        target.validate_probe_recovery(receipt, plan)
    receipt = probe_receipt(plan)
    receipt['container'] = 'v06-train-state-v1'
    with pytest.raises(target.ProbeError):
        target.validate_probe_recovery(receipt, plan)
    receipt = probe_receipt(plan)
    receipt['counts']['counts_against_training'] = True
    with pytest.raises(target.ProbeError):
        target.validate_probe_recovery(receipt, plan)


def test_probe_ledger_records_56_updates_and_one_recovery():
    plan = target.probe_plan()
    ledger = target.ProbeLedger(plan)
    for index in range(1, 56):
        ledger.record_update(index)
    with pytest.raises(target.ProbeError):
        ledger.record_update(55)  # never billed twice
    resume = ledger.resume(56)
    assert resume['resumed_from_update'] == 55
    assert resume['updates_after'] == 1
    assert resume['billed_updates_not_replayed'] is True
    ledger.record_update(56)
    receipt = ledger.receipt()
    assert receipt['updates'] == 56
    assert receipt['updates_after_recovery'] == 1
    with pytest.raises(target.ProbeError):
        ledger.resume(57)
    with pytest.raises(target.ProbeError):
        target.ProbeLedger(plan).resume(55)


# ------------------------------------- Q-fix: probe entry point wiring
def test_main_accepts_a_probe_factory_and_help_exits_zero(tmp_path):
    import inspect
    signature = inspect.signature(target.main)
    assert 'probe_factory' in signature.parameters
    assert 'probe_factory' in inspect.signature(
        target.V06Orchestrator.__init__).parameters
    with pytest.raises(SystemExit) as exit_info:
        target.main(['--mode', 'probe', '--help'])
    assert exit_info.value.code == 0


def test_main_probe_without_execute_only_preflights(tmp_path):
    root = probe_root(tmp_path)
    code = target.main(['--mode', 'probe'], root=root)
    assert code == 0
    assert not (root / 'logs/a2_ideal_stairs_v06/implementation/probe').exists()
    preflight = root / 'logs/a2_ideal_stairs_v06/implementation/preflight'
    assert (preflight / 'preflight_probe_seed42.json').is_file()
    assert not (root / 'src').exists()


def test_main_probe_execute_requires_a_probe_worker(tmp_path):
    root = probe_root(tmp_path)
    args = ['--mode', 'probe', '--execute',
            '--output-dir',
            str(root / 'logs/a2_ideal_stairs_v06/implementation/probe/probe_seed42')]
    with pytest.raises(target.AdmissionError) as error:
        target.main(args, root=root,
                    resource_probe=lambda directory=None: good_snapshot(),
                    worker_factory=lambda contract, directory: {})
    assert 'probe worker' in str(error.value)


def test_run_script_wires_the_probe_factory():
    import run_a2_ideal_stairs_v06 as run_script
    import train_a2_ideal_stairs_v06 as train_script
    import inspect

    assert callable(train_script.build_probe_worker)
    source = inspect.getsource(run_script.main)
    assert 'probe_factory' in source and 'build_probe_worker' in source


# ------------------------------------- probe cumulative counts (55 + 1 = 56)
def probe_learning(update, *, num_envs=512, rollout=48):
    controls = int(update) * int(num_envs) * int(rollout)
    return dict(update=int(update), optimizer_step=int(update) * 20,
                adam_steps=int(update) * 20, control_samples=controls,
                physics_substeps=4 * controls, learning_rate=3e-4)


def test_probe_counts_combine_checkpoint_and_recovery():
    import train_a2_ideal_stairs_v06 as train
    from src.tasks.velocity.v06 import orchestration as orchestration_module

    plan = target.probe_plan()
    before = probe_learning(55)
    after = probe_learning(56)
    counts, evidence = train.probe_counts(before, after, plan)
    assert counts['learning_updates'] == 56
    assert counts['control_samples'] == 1_376_256
    assert counts['physics_substeps'] == 5_505_024
    assert counts['adam_updates'] == 1_120
    assert counts['counts_against_training'] is False
    assert counts['training_budget_untouched'] == 30000
    assert counts['training_budget_consumed'] == 0
    assert evidence['delta'] == dict(control_samples=24_576,
                                     physics_substeps=98_304, adam_updates=20)
    assert evidence['before'] == 55 and evidence['after'] == 56
    # The receipt satisfies the frozen recovery contract.
    receipt = dict(resume_mode='new_episode', resumed_from_update=55,
                   updates_after=1, billed_updates_not_replayed=True,
                   initialized_from_eth_again=False,
                   container=target.probe_container(),
                   checkpoint_sha256='a' * 64,
                   curriculum_state={'phase': 'R'}, counts=counts)
    assert orchestration_module.validate_probe_recovery(receipt, plan) is True


def test_probe_counts_reject_bad_increments_and_missing_fields():
    import train_a2_ideal_stairs_v06 as train
    from src.tasks.velocity.v06 import orchestration as orchestration_module

    plan = target.probe_plan()
    with pytest.raises(orchestration_module.ProbeError):
        # Only one update in the resumed block: the total would be 1, not 56.
        train.probe_counts(probe_learning(0), probe_learning(1), plan)
    with pytest.raises(orchestration_module.ProbeError):
        train.probe_counts(probe_learning(55), probe_learning(57), plan)
    broken = probe_learning(55)
    del broken['control_samples']
    with pytest.raises(orchestration_module.ProbeError):
        train.probe_counts(broken, probe_learning(56), plan)
    # A wrong single-update delta (as if a replayed update were billed) is refused.
    replay = probe_learning(57)
    with pytest.raises(orchestration_module.ProbeError):
        train.probe_counts(probe_learning(55), replay, plan)
    # A cumulative total that does not match the frozen cost is refused.
    short = probe_learning(55)
    short['control_samples'] = 1
    with pytest.raises(orchestration_module.ProbeError):
        train.probe_counts(short, probe_learning(56), plan)


def test_probe_worker_uses_checkpoint_cumulative_counts():
    import inspect
    import train_a2_ideal_stairs_v06 as train

    source = inspect.getsource(train.build_probe_worker)
    assert "state['learning']" in source
    assert 'probe_counts(' in source
    assert 'counts_at_checkpoint' in source
    assert 'single_update_delta' in source


# ------------------------------------- process MiB provenance (compute apps/torch)
def test_resource_snapshot_matches_this_process_compute_apps(tmp_path):
    import os
    snapshot = target.resource_snapshot(
        directory=tmp_path,
        query=[dict(index=0, used_mib=1000, total_mib=24000)],
        processes=[dict(pid=os.getpid(), used_mib=1234),
                   dict(pid=os.getpid() + 1, used_mib=99)])
    assert snapshot['whole_device_mib'] == 1000
    assert snapshot['process_mib'] == 1234
    assert snapshot['process_mib_method'] == 'compute_apps'
    assert snapshot['compute_apps'][0]['pid'] == os.getpid()


def test_resource_snapshot_falls_back_to_torch_reserved(tmp_path):
    snapshot = target.resource_snapshot(
        directory=tmp_path,
        query=[dict(index=0, used_mib=1000, total_mib=24000)],
        processes=[], torch_reserved=2345.4)
    assert snapshot['process_mib'] == 2345
    assert snapshot['process_mib_method'] == 'torch_reserved'
    # Without compute-apps or a CUDA reservation the value stays honestly None.
    empty = target.resource_snapshot(
        directory=tmp_path,
        query=[dict(index=0, used_mib=1000, total_mib=24000)],
        processes=[], torch_reserved=None)
    assert empty['process_mib'] is None
    assert empty['process_mib_method'] is None
    guarded = target.check_resources(empty)
    assert guarded['ok'] is False
    assert 'process_mib' in guarded['blocking']


def test_nvidia_smi_parses_gpu_and_compute_apps_tables(monkeypatch):
    class Result:
        def __init__(self, stdout):
            self.stdout = stdout

    def fake_run(command, **kwargs):
        if '--query-compute-apps=pid,used_memory' in command:
            return Result('12345, 4100\n23456, 100\nnot-a-pid, 5\n')
        return Result('0, 4316, 24576\n')

    monkeypatch.setattr(target.subprocess, 'run', fake_run)
    tables = target._nvidia_smi()
    assert tables['gpus'] == [dict(index=0, used_mib=4316, total_mib=24576)]
    assert tables['processes'] == [dict(pid=12345, used_mib=4100),
                                   dict(pid=23456, used_mib=100)]
    snapshot = target.resource_snapshot(directory=None, query=None, processes=None,
                                        pid=12345)
    assert snapshot['whole_device_mib'] == 4316
    assert snapshot['process_mib'] == 4100
    assert snapshot['process_mib_method'] == 'compute_apps'
    # A PID outside the tree falls back (torch unavailable in CPU tests) and is
    # reported as unavailable rather than a silent zero.
    missing = target.resource_snapshot(directory=None, query=tables['gpus'],
                                       processes=tables['processes'], pid=999999,
                                       torch_reserved=None)
    assert missing['process_mib'] is None
    assert missing['process_mib_method'] is None


def test_resource_summary_keeps_the_process_method():
    samples = probe_samples(process_mib_method='torch_reserved')
    summary = target.summarize_resource_samples(samples)
    assert summary['metrics']['process_mib']['method'] == 'torch_reserved'
    mixed = probe_samples()
    mixed[0]['process_mib_method'] = 'torch_reserved'
    assert target.summarize_resource_samples(mixed)['metrics']['process_mib']['method'] \
        == 'mixed'


# ------------------------------------- pre-run vs measured resource stages
def test_pre_run_allows_uninitialized_process_but_measured_does_not(tmp_path):
    # Before CUDA init the process legitimately owns no allocation.
    pre = target.resource_snapshot(
        directory=tmp_path,
        query=[dict(index=0, used_mib=4316, total_mib=24576)],
        processes=[], torch_reserved=None, stage='pre_run')
    assert pre['process_mib'] is None
    assert pre['process_mib_method'] == 'not_initialized'
    allowed = target.check_resources(pre, stage='pre_run')
    assert allowed['ok'] is True
    assert allowed['exempted'] == ['process_mib']
    assert allowed['checks']['process_mib']['exempted'] is True
    assert allowed['checks']['process_mib']['method'] == 'not_initialized'
    # The same evidence inside the measured window stays fail-closed.
    measured = target.resource_snapshot(
        directory=tmp_path,
        query=[dict(index=0, used_mib=4316, total_mib=24576)],
        processes=[], torch_reserved=None, stage='measured')
    assert measured['process_mib'] is None
    assert measured['process_mib_method'] is None
    blocked = target.check_resources(measured, stage='measured')
    assert blocked['ok'] is False
    assert blocked['blocking'] == ['process_mib']


def test_measured_process_memory_still_obeys_the_5120_limit(tmp_path):
    ok = target.resource_snapshot(
        directory=tmp_path,
        query=[dict(index=0, used_mib=4316, total_mib=24576)],
        processes=[dict(pid=__import__('os').getpid(), used_mib=4000)])
    result = target.check_resources(ok, stage='measured')
    assert result['ok'] is True and result['exempted'] == []
    heavy = target.resource_snapshot(
        directory=tmp_path,
        query=[dict(index=0, used_mib=4316, total_mib=24576)],
        processes=[dict(pid=__import__('os').getpid(), used_mib=6000)])
    refused = target.check_resources(heavy, stage='measured')
    assert refused['ok'] is False
    assert refused['blocking'] == ['process_mib']
    assert refused['checks']['process_mib']['limit'] == 5120


def test_probe_pre_run_exemption_still_runs_the_probe(tmp_path):
    def pre_snapshot(directory=None):
        return dict(timestamp=0.0, device=0, whole_device_mib=4316,
                    process_mib=None, process_mib_method=None,
                    host_available_mib=17000, disk_free_gib=450.0)

    def factory(plan, output):
        return dict(cost=dict(plan['cost']),
                    resources=dict(samples=probe_samples()),
                    recovery=probe_receipt(plan), phase_timeline=[])

    orchestrator = target.V06Orchestrator(
        probe_args(tmp_path), root=probe_root(tmp_path),
        preflight_dir=tmp_path / 'preflight', resource_probe=pre_snapshot,
        probe_factory=factory)
    orchestrator.preflight()
    result = orchestrator.execute()
    assert result['status'] == 'engineering_check_only'
    pre = result['probe']['resources']['pre_run_check']
    assert pre['exempted'] == ['process_mib']
    assert pre['checks']['process_mib']['method'] == 'not_initialized'
    assert result['probe']['status'] == 'engineering_check_only'
    assert result['probe']['resources']['measured']['metrics']['process_mib']['ok'] is True


def test_probe_measured_process_none_is_resource_insufficient(tmp_path):
    def factory(plan, output):
        samples = probe_samples()
        for sample in samples:
            sample['process_mib'] = None
            sample['process_mib_method'] = None
        return dict(cost=dict(plan['cost']),
                    resources=dict(samples=samples),
                    recovery=probe_receipt(plan), phase_timeline=[])

    orchestrator = target.V06Orchestrator(
        probe_args(tmp_path), root=probe_root(tmp_path),
        preflight_dir=tmp_path / 'preflight',
        resource_probe=lambda directory=None: good_snapshot(),
        probe_factory=factory)
    orchestrator.preflight()
    result = orchestrator.execute()
    assert result['status'] == 'resource_insufficient'
    assert result['probe']['status'] == 'resource_insufficient'
    assert 'process_mib' in result['probe']['resources']['blocking']


# ------------------------------------- train admission: pre-run vs measured
def test_train_admission_pre_run_exempts_the_uninitialized_process_memory(tmp_path,
                                                                        monkeypatch):
    def pre_snapshot(directory=None):
        return dict(timestamp=0.0, device=0, whole_device_mib=4316,
                    process_mib=None, process_mib_method=None,
                    host_available_mib=17000, disk_free_gib=450.0)

    def worker(contract, output):
        return {'state': 'running'}

    orchestrator = target.V06Orchestrator(
        execute_args(tmp_path), preflight_dir=tmp_path / 'preflight',
        resource_probe=pre_snapshot, worker_factory=worker)
    monkeypatch.setattr(target, 'validate_arguments',
                        partial(_real_validate, require_acceptance=False))
    orchestrator.preflight()
    result = orchestrator.execute()
    assert result['status'] == 'running'
    pre = result['resources']
    assert pre['stage'] == 'pre_run'
    assert pre['exempted'] == ['process_mib']
    assert pre['checks']['process_mib']['ok'] is True
    evidence = json.loads((Path(result['contract']['output_dir']) / 'resources'
                           / 'pre_run.json').read_text(encoding='utf-8'))
    assert evidence['stage'] == 'pre_run'
    assert evidence['exempted'] == ['process_mib']
    assert evidence['process_mib_method'] == 'not_initialized'


def test_train_admission_still_enforces_whole_card_host_and_disk(tmp_path, monkeypatch):
    def pre_snapshot(directory=None):
        return dict(timestamp=0.0, device=0, whole_device_mib=9000,
                    process_mib=None, process_mib_method=None,
                    host_available_mib=17000, disk_free_gib=450.0)

    orchestrator = target.V06Orchestrator(
        execute_args(tmp_path), preflight_dir=tmp_path / 'preflight',
        resource_probe=pre_snapshot,
        worker_factory=lambda contract, output: {'state': 'running'})
    monkeypatch.setattr(target, 'validate_arguments',
                        partial(_real_validate, require_acceptance=False))
    with pytest.raises(target.AdmissionError) as error:
        orchestrator.execute()
    assert 'whole_device_mib' in str(error.value)
    assert not (tmp_path / 'run').exists()


def test_worker_measured_guard_blocks_when_process_memory_is_missing(tmp_path,
                                                                    monkeypatch):
    import train_a2_ideal_stairs_v06 as train

    holder = {'runner': type('R', (), {'update': 1})()}
    guard = train.build_measured_resource_guard(tmp_path, 'cuda:0', holder)

    def missing(directory=None, device=0):
        return dict(timestamp=0.0, device=0, whole_device_mib=4316,
                    process_mib=None, process_mib_method=None,
                    host_available_mib=17000, disk_free_gib=450.0)

    monkeypatch.setattr(target, 'resource_snapshot', missing)
    result = guard()
    assert result['stage'] == 'measured'
    assert result['ok'] is False
    assert result['blocking'] == ['process_mib']
    assert result['update'] == 1
    evidence = json.loads((tmp_path / 'resources' / 'measured_first_update.json')
                          .read_text(encoding='utf-8'))
    assert evidence['check']['blocking'] == ['process_mib']
    # Cached: the measured sample is taken exactly once.
    calls = []
    monkeypatch.setattr(target, 'resource_snapshot',
                        lambda **kwargs: calls.append(kwargs) or missing())
    assert guard()['ok'] is False
    assert calls == []


@pytest.mark.parametrize('process_mib,expected_ok', [(4000, True), (6000, False)])
def test_worker_measured_guard_enforces_the_5120_limit(tmp_path, monkeypatch,
                                                       process_mib, expected_ok):
    import os
    import train_a2_ideal_stairs_v06 as train

    holder = {'runner': type('R', (), {'update': 1})()}
    guard = train.build_measured_resource_guard(tmp_path, 'cuda:0', holder)

    def snapshot(directory=None, device=0):
        return dict(timestamp=0.0, device=device, whole_device_mib=4316,
                    process_mib=process_mib, process_mib_method='compute_apps',
                    host_available_mib=17000, disk_free_gib=450.0,
                    gpu_query=[], compute_apps=[], stage='measured')

    monkeypatch.setattr(target, 'resource_snapshot', snapshot)
    result = guard()
    assert result['ok'] is expected_ok
    if not expected_ok:
        assert result['blocking'] == ['process_mib']


def test_train_worker_wires_the_measured_resource_guard():
    import inspect
    import train_a2_ideal_stairs_v06 as train

    source = inspect.getsource(train.build_worker)
    assert 'build_measured_resource_guard' in source
    assert 'resource_guard=resource_guard' in source
    guard_source = inspect.getsource(train.build_measured_resource_guard)
    assert "stage='measured'" in guard_source
    assert 'measured_first_update.json' in guard_source


def _bridge_checkpoint(tmp_path, monkeypatch):
    from src.tasks.velocity.v06 import recovery as recovery_module
    from test_v06_schema import make_state
    state = make_state(update=100, phase='F0')
    state['ledger']['pending_boundary'] = dict(
        update=100, phase='F0', substage=None, generation=0,
        tasks=[dict(kind='validation'), dict(kind='diagnostic')],
        results=[], curriculum_decided=False, committed=True)
    path = tmp_path / 'boundary_000100.pt'
    torch.save(state, path)
    actual = recovery_module.sha256(path)
    monkeypatch.setattr(recovery_module, 'BRIDGE_CHECKPOINT_SHA256', actual)
    monkeypatch.setattr(recovery_module, 'BRIDGE_OLD_SOURCE_CLOSURE_SHA256',
                        state['source']['source_closure_sha256'])
    root = tmp_path / 'root'
    manifest = (root / 'logs/a2_ideal_stairs_v06/implementation/coordination'
                / 'Q005_recovery_bridge.md')
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(f'{actual}\n{state["source"]["source_closure_sha256"]}\n'
                        f'{recovery_module.Q005_DECISION_SHA256}\n', encoding='utf-8')
    return path, state, recovery_module, root


def test_q005_bridge_flag_required_and_validated(tmp_path, monkeypatch):
    path, state, recovery_module, root = _bridge_checkpoint(tmp_path, monkeypatch)
    base = ['--mode', 'train', '--resume-run', str(path),
            '--output-dir', str(tmp_path / 'run')]
    argv = base + ['--allow-q005-bridge']
    orchestrator = target.V06Orchestrator(
        target.build_parser().parse_args(argv), root=root,
        preflight_dir=tmp_path / 'preflight')
    record = orchestrator.preflight()
    assert record['contract']['allow_q005_bridge'] is True
    assert record['resume']['bridge']['restricted'] is True
    assert record['resume']['learning_rate'] == pytest.approx(3e-4)
    assert record['resume']['pending_boundary_requires_completion'] is True

    # Without the restricted flag the documented checkpoint is refused.
    plain = target.V06Orchestrator(
        target.build_parser().parse_args(base), root=root,
        preflight_dir=tmp_path / 'preflight2')
    with pytest.raises(target.PreflightError):
        plain.preflight()
    # Any other source is refused even with the flag.
    monkeypatch.setattr(recovery_module, 'BRIDGE_CHECKPOINT_SHA256', 'a' * 64)
    other = target.V06Orchestrator(
        target.build_parser().parse_args(argv), root=root,
        preflight_dir=tmp_path / 'preflight3')
    with pytest.raises(target.PreflightError):
        other.preflight()


def test_q005_manifest_mismatch_and_generic_closure_check(tmp_path, monkeypatch):
    import train_a2_ideal_stairs_v06 as train
    from test_v06_schema import make_state
    from src.tasks.velocity.v06 import recovery as recovery_module

    manifest = tmp_path / 'manifest.md'
    manifest.write_text('placeholder', encoding='utf-8')
    contract = dict(resume=dict(checkpoint_sha256=recovery_module.BRIDGE_CHECKPOINT_SHA256))
    with pytest.raises(SystemExit):
        train.verify_q005_bridge(contract, manifest=manifest)
    manifest.write_text(
        f"{recovery_module.BRIDGE_CHECKPOINT_SHA256}\n"
        f"{recovery_module.BRIDGE_OLD_SOURCE_CLOSURE_SHA256}\n"
        f"{recovery_module.Q005_DECISION_SHA256}\n", encoding='utf-8')
    record = train.verify_q005_bridge(contract, manifest=manifest)
    assert record['manifest_sha256'] == recovery_module.sha256(manifest)
    # Ordinary resume still refuses a checkpoint whose source closure differs.
    state = make_state()
    state['source']['source_closure_sha256'] = 'b' * 64
    path = tmp_path / 'foreign.pt'
    torch.save(state, path)
    with pytest.raises(recovery_module.RecoveryBlocked):
        recovery_module.preload_checkpoint(path, mode='resume')
