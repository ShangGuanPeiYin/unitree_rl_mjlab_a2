"""CPU mock resource protocol tests. No CUDA model or simulator is created."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from a2_v07_bootstrap import activate
activate(ROOT)
from src.tasks.velocity.v09 import spec, resources as r
from src.tasks.velocity.v09.cases import sha256


@pytest.mark.parametrize('value', [
    'GPU-5fd3145a-6ae6-4777-33c0-bb1a32cac8b9',
    '5fd3145a-6ae6-4777-33c0-bb1a32cac8b9',
    'GPU-5FD3145A-6AE6-4777-33C0-BB1A32CAC8B9',
    b'GPU-5fd3145a-6ae6-4777-33c0-bb1a32cac8b9',
])
def test_gpu_uuid_representation_has_one_canonical_identity(value):
    assert r.canonical_gpu_uuid(value) == '5fd3145a-6ae6-4777-33c0-bb1a32cac8b9'


def _mock_uuid_backend(monkeypatch, cuda_uuid, memory_state=None):
    class Nvml:
        def nvmlInit_v2(self):
            return 0
        def nvmlDeviceGetHandleByIndex_v2(self, index, handle):
            handle._obj.value = 1
            return 0
        def nvmlDeviceGetUUID(self, handle, buffer, size):
            buffer.value = b'GPU-5fd3145a-6ae6-4777-33c0-bb1a32cac8b9'
            return 0
        def nvmlDeviceGetMemoryInfo(self, handle, memory):
            memory._obj.total = 12 * 1024**3
            memory._obj.used = memory_state['used']
            memory._obj.free = memory._obj.total - memory_state['used']
            return 0
    monkeypatch.setattr(r.ctypes, 'CDLL', lambda path: Nvml())
    monkeypatch.setattr(r.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(r.torch.cuda, 'get_device_properties', lambda device: SimpleNamespace(uuid=cuda_uuid))


def test_monitor_accepts_same_uuid_object_without_nvml_gpu_prefix(monkeypatch):
    class CudaUuid:
        def __str__(self):
            return '5fd3145a-6ae6-4777-33c0-bb1a32cac8b9'
    _mock_uuid_backend(monkeypatch, CudaUuid())
    monitor = r.WholeCardMonitor(device='cuda:0', physical_gpu_index=0, interval=.02)
    assert monitor.gpu_uuid == 'GPU-5fd3145a-6ae6-4777-33c0-bb1a32cac8b9'
    assert monitor.samples == 0  # identity-only CPU test, no physical/GPU sampling


def test_monitor_rejects_a_different_gpu_even_with_valid_uuid_format(monkeypatch):
    _mock_uuid_backend(monkeypatch, '5fd3145a-6ae6-4777-33c0-bb1a32cac8b8')
    with pytest.raises(RuntimeError, match='UUID mismatch'):
        r.WholeCardMonitor(device='cuda:0', physical_gpu_index=0, interval=.02)


@pytest.mark.parametrize('value', [None, '', 'GPU-not-a-uuid', 'MIG-5fd3145a-6ae6-4777-33c0-bb1a32cac8b9'])
def test_gpu_uuid_invalid_identity_is_rejected(value):
    with pytest.raises((TypeError, ValueError)):
        r.canonical_gpu_uuid(value)


def test_q009_phase_peaks_do_not_add_exclusive_training_and_child_memory(monkeypatch):
    gib = 1024**3
    state = dict(used=6 * gib, reserved=6 * gib, peak=6 * gib)
    _mock_uuid_backend(monkeypatch, '5fd3145a-6ae6-4777-33c0-bb1a32cac8b9', state)
    monkeypatch.setattr(r.torch.cuda, 'synchronize', lambda device: None)
    monkeypatch.setattr(r.torch.cuda, 'mem_get_info', lambda device: (12 * gib - state['used'], 12 * gib))
    monkeypatch.setattr(r.torch.cuda, 'memory_reserved', lambda device: state['reserved'])
    monkeypatch.setattr(r.torch.cuda, 'max_memory_reserved', lambda device: state['peak'])
    monkeypatch.setattr(r.torch.cuda, 'reset_peak_memory_stats', lambda device: state.update(peak=state['reserved']))
    monkeypatch.setattr(r, 'host_available_bytes', lambda: 8 * gib)
    monitor = r.WholeCardMonitor(device='cuda:0', physical_gpu_index=0, interval=.02).start()
    state.update(used=0, reserved=0)
    monitor.begin_phase('unloaded_child_evaluation')
    state.update(used=3 * gib)
    monitor.observe()
    result = monitor.finish()
    assert result['conservative_whole_card_peak_bytes'] == 6 * gib
    assert [phase['conservative_whole_card_peak_bytes'] for phase in result['phases']] == [6 * gib, 3 * gib]


class Ledger:
    def __init__(self):
        self.calls, self.operations = [], {}
        self.duties = {f'probe_{n}': dict(spent=0, closed=False) for n in spec.ENV_CANDIDATES}

    def reserve(self, **kwargs):
        key = kwargs['operation_id']
        assert key not in self.operations
        assert kwargs['course'] == 'S1_RESOURCE'
        self.operations[key] = dict(kwargs, settled=False)
        self.calls.append(('reserve', key))
        return key

    def commit(self, operation, *, actual_control_steps=None, status='complete'):
        row = self.operations[operation]
        assert not row['settled']
        row.update(settled=True, status=status,
                   actual_control_steps=row['max_control_steps'] if status == 'unknown' else actual_control_steps)
        if status == 'complete':
            assert actual_control_steps == row['max_control_steps']
        self.duties[row['obligation_id']]['spent'] += row['num_envs'] * row['actual_control_steps']
        self.calls.append(('commit', operation, status))

    def close_obligation(self, stage, duty, **kwargs):
        self.duties[duty]['closed'] = True
        self.calls.append(('close', duty))

    def fail_probe_obligation(self, stage, duty, **kwargs):
        assert len(kwargs['evidence_sha256']) == 64
        self.duties[duty]['failed'] = True
        self.calls.append(('fail', duty))

    def skip_unattempted_probe(self, stage, duty, **kwargs):
        assert not self.duties[duty]['spent']
        self.duties[duty]['skipped'] = True
        self.calls.append(('skip', duty))

    def snapshot(self):
        return {'stages': {'S1': {'obligations': self.duties}}}

    def checkpoint_metadata(self):
        return {'ledger_run_id': 'mock', 'spent': sum(d['spent'] for d in self.duties.values())}


class Monitor:
    def __init__(self, peak=r.GPU_LIMIT_BYTES, host=r.HOST_MIN_BYTES):
        self.peak, self.host = peak, host
        self.samples = 0
        self.phases = ['training']
    def start(self):
        return self
    def observe(self):
        self.samples += 1
    def begin_phase(self, name):
        self.phases.append(name)
    def finish(self):
        return dict(monitoring_valid=True, conservative_whole_card_peak_bytes=self.peak,
                    host_min_available_bytes=self.host, samples=self.samples,
                    phases=[{'name': name} for name in self.phases])


class Worker:
    def __init__(self, ledger, n, *, failure=None, eval_steps=5005, failure_error=MemoryError):
        self.ledger, self.n, self.failure, self.eval_steps = ledger, n, failure, eval_steps
        self.indices, self.closed = [], False
        self.failure_error = failure_error
        self.identity = {'source_sha256': spec.SOURCE_SHA256}
        self.offloaded = False
        self.external_modes = []
    def iteration(self, index):
        item = f'rollout_{index + 1:03d}' if index < 55 else 'extra_001' if index == 55 else 'extra_004'
        key = f'n{self.n}_{item}'
        assert key in self.ledger.operations and not self.ledger.operations[key]['settled']
        self.indices.append(index)
        if self.failure == index:
            raise self.failure_error('failure within partly advanced rollout')
        return dict(control_steps=48, adam_steps=20, samples=self.n * 48,
                    loss=.1, action_loss=.05, value_loss=.5, gradient_norm=.2,
                    resource_course='S1.1' if index < 55 else 'S1.2',
                    trainable_parameter_count=18 if index < 55 else 44,
                    adam_allocated_parameter_count=18 if index < 55 else 44,
                    full_parameter_adam_names=None if index < 55 else [f'p{i}' for i in range(44)])
    def save_and_restore(self):
        assert f'n{self.n}_extra_001' in self.ledger.operations
        return {'optimizer_restored': True, 'checkpoint_sha256': 'a' * 64}
    def offload_for_evidence(self):
        assert self.ledger.operations[f'n{self.n}_extra_001']['settled']
        self.offloaded = True
        return {'checkpoint_sha256': 'e' * 64}
    def external_assessment(self, mode, operation):
        assert self.offloaded
        expected = f'n{self.n}_extra_002' if mode == 'evaluation' else f'n{self.n}_extra_003'
        assert operation == expected and not self.ledger.operations[operation]['settled']
        self.external_modes.append(mode)
        return dict(control_steps=self.eval_steps if mode == 'evaluation' else 5005,
                    num_envs=100 if mode == 'evaluation' else 1,
                    padding_charged=True, behavior_gate=False,
                    video_verified=True, terminal_frames_pre_reset=True)
    def restore_after_evidence(self):
        assert self.offloaded and self.external_modes == ['evaluation', 'video']
        assert f'n{self.n}_extra_004' in self.ledger.operations
        self.offloaded = False
        return dict(fingerprints_verified=True, rng_restored=True)
    def close(self):
        self.closed = True


def run(tmp_path, *, fail_first=None, first_eval_steps=5005, high_memory=False, engineering_failure=False):
    ledger, workers, order = Ledger(), {}, []
    def factory(n, path):
        order.append(n)
        worker = Worker(ledger, n, failure=fail_first if n == 2048 else None,
                        eval_steps=first_eval_steps if n == 2048 else 5005,
                        failure_error=RuntimeError if engineering_failure else MemoryError)
        workers[n] = worker
        return worker
    report = r.run_resource_probe(ledger=ledger, segment_id='segment', calibration={'value_scale': 2.},
        output_dir=tmp_path / 'probe', code_sha256='c' * 64, candidate_factory=factory,
        monitor_factory=lambda n: Monitor(peak=r.GPU_LIMIT_BYTES + 1 if high_memory else r.GPU_LIMIT_BYTES))
    return report, ledger, workers, order


def test_selects_largest_after_all55_full_iterations_recovery_and_padding(tmp_path):
    report, ledger, workers, order = run(tmp_path)
    assert order == [2048]
    assert report['admitted'] and report['selected_num_envs'] == 2048
    worker = workers[2048]
    assert worker.indices == list(range(57)) and worker.closed
    result = report['candidates'][0]
    assert result['warmups_completed'] == 5 and result['measurements_completed'] == 50
    assert result['restored_iteration_completed'] and result['evaluation_control_steps'] == 5005
    assert len(ledger.operations) == 59
    assert ledger.duties['probe_2048']['spent'] == 2048 * 57 * 48 + 101 * 5005
    assert result['evaluation_num_envs'] == 100 and result['video_control_steps'] == 5005
    assert result['post_evidence_iteration_completed']
    assert ledger.duties['probe_2048']['closed']
    assert all(ledger.duties[f'probe_{n}']['skipped'] for n in (1536, 1024, 512))
    payload = {key: value for key, value in report.items() if key != 'report_sha256'}
    assert report['report_sha256'] == sha256(payload)


def test_partial_failure_bills_unknown_full_bound_and_never_claims55(tmp_path):
    report, ledger, workers, order = run(tmp_path, fail_first=2)
    assert order == [2048, 1536]
    failed = report['candidates'][0]
    assert failed['warmups_completed'] == 2 and failed['measurements_completed'] == 0
    assert not failed['admitted'] and not failed['restored_iteration_completed']
    assert ledger.operations['n2048_rollout_003']['status'] == 'unknown'
    assert ledger.duties['probe_2048']['spent'] == 2048 * 3 * 48
    assert ledger.duties['probe_2048']['failed']
    assert report['selected_num_envs'] == 1536


def test_eval_early_return_is_not_complete_and_is_billed_full5005(tmp_path):
    report, ledger, workers, order = run(tmp_path, first_eval_steps=4999)
    assert order == [2048]
    first = report['candidates'][0]
    assert first['evaluation_control_steps'] == 0 and not first['admitted']
    assert ledger.operations['n2048_extra_002']['status'] == 'unknown'
    assert ledger.operations['n2048_extra_002']['actual_control_steps'] == 5005
    assert report['halted_for_engineering'] and report['selected_num_envs'] is None


def test_ordinary_runtime_consistency_error_halts_without_lowering_N(tmp_path):
    report, ledger, workers, order = run(tmp_path, fail_first=2, engineering_failure=True)
    assert order == [2048]
    assert report['halted_for_engineering'] and not report['admitted']
    assert report['candidates'][0]['error']['type'] == 'RuntimeError'
    assert ledger.operations['n2048_rollout_003']['status'] == 'unknown'


def test_all_four_memory_failures_do_not_invent_lower_candidate_or_admission(tmp_path):
    report, ledger, workers, order = run(tmp_path, high_memory=True)
    assert order == [2048, 1536, 1024, 512]
    assert not report['admitted'] and report['selected_num_envs'] is None
    assert all(ledger.duties[f'probe_{n}']['failed'] for n in order)


def test_exact_gpu_host_limits_and_missing_measurement_fail_closed():
    monitor = Monitor()
    for name in ('full_s1_2', 'unloaded_child_evaluation', 'unloaded_child_video', 'resume_training'):
        monitor.begin_phase(name)
    result = dict(warmups_completed=5, measurements_completed=50,
                  restored_iteration_completed=True, evaluation_control_steps=5005,
                  full_s1_2_workload_verified=True,
                  evaluation_num_envs=100, video_control_steps=5005, post_evidence_iteration_completed=True,
                  error=None, memory=monitor.finish())
    assert r.qualifies(result)
    result['full_s1_2_workload_verified'] = False
    assert not r.qualifies(result)
    result['full_s1_2_workload_verified'] = True
    result['memory']['host_min_available_bytes'] -= 1
    assert not r.qualifies(result)
    result['memory']['host_min_available_bytes'] += 1
    result['measurements_completed'] = 49
    assert not r.qualifies(result)


def test_calibration_never_supplies_default_scale_or_final_data(tmp_path):
    data = dict(sds_sha256=spec.SDS_SHA256, source_sha256=spec.SOURCE_SHA256,
                manifest_sha256='a' * 64, code_sha256='b' * 64, value_scale=2.,
                value_samples=1000, physical_control_samples=1000, method='population_std_max_1')
    path = tmp_path / 'calibration.json'
    path.write_text(json.dumps(data))
    result = r.load_calibration(path, code_sha256='b' * 64)
    assert result['calibration_file_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    for key, value in (('value_scale', float('nan')), ('value_scale', .9), ('method', 'final_statistics'),
                       ('code_sha256', 'c' * 64), ('physical_control_samples', 1_000_001)):
        path.write_text(json.dumps(dict(data, **{key: value})))
        with pytest.raises(ValueError):
            r.load_calibration(path, code_sha256='b' * 64)
    del data['value_scale']
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        r.load_calibration(path, code_sha256='b' * 64)


def calibration_context(tmp_path):
    from v09_identity_fixtures import make_identity_fixture
    from src.tasks.velocity.v09.evidence_identity import EvidenceIdentityContext
    fixture = make_identity_fixture(tmp_path, consumer_code_sha256='c' * 64)
    context = EvidenceIdentityContext.from_approval(fixture.approval_path, fixture.approval_sha256,
                                                    **fixture.context_kwargs())
    return fixture, context


def test_approved_legacy_calibration_keeps_producer_values_and_verified_ref(tmp_path):
    fixture, context = calibration_context(tmp_path)
    original = fixture.calibration_path.read_bytes()
    result = r.load_calibration(fixture.calibration_path, code_sha256='c' * 64, identity_context=context)
    assert result['code_sha256'] == '44780cd5d6f9ee4029fb38b2f9348a872268e7bedf995d4b36e2e30ec2f2163f'
    assert result['value_scale'] == 2. and result['value_samples'] == 600
    assert result['physical_control_samples'] == 720
    assert result['calibration_file_sha256'] == hashlib.sha256(original).hexdigest()
    assert result['evidence_ref']['producer_code_sha256'] == result['code_sha256']
    assert result['evidence_ref']['consumer_code_sha256'] == 'c' * 64
    assert result['evidence_ref']['approval_sha256'] == fixture.approval_sha256
    assert result['calibration_path'] == result['evidence_ref']['path']
    assert fixture.calibration_path.read_bytes() == original


def test_legacy_calibration_without_context_remains_rejected(tmp_path):
    from v09_identity_fixtures import make_identity_fixture
    fixture = make_identity_fixture(tmp_path)
    with pytest.raises(ValueError):
        r.load_calibration(fixture.calibration_path, code_sha256='c' * 64, identity_context=None)


def test_calibration_context_cannot_authorize_a_different_consumer(tmp_path):
    fixture, context = calibration_context(tmp_path)
    with pytest.raises(ValueError):
        r.load_calibration(fixture.calibration_path, code_sha256='d' * 64, identity_context=context)


def test_legacy_calibration_changed_after_approval_is_rejected(tmp_path):
    fixture, context = calibration_context(tmp_path)
    fixture.put_json(fixture.calibration_path, dict(fixture.calibration, value_scale=3.))
    with pytest.raises(ValueError):
        r.load_calibration(fixture.calibration_path, code_sha256='c' * 64, identity_context=context)


def test_role_allocation_and_provider_keep_native_frozen_case_inputs():
    roles = r.role_allocation(2048)
    assert [roles.count(role) for role in ('flat_walk', 'flat_stop', 'H00_H15', 'D25_D22_D20_L12_L20_L40')] == [410, 410, 614, 614]
    provider = r.training_provider('resource_2048')
    for role in set(roles):
        assert provider(role, 9) == provider(role, 9)
        assert provider(role, 9)['streams'] != provider(role, 10)['streams']


def test_admission_report_write_failure_does_not_close_or_skip_paid_work(tmp_path, monkeypatch):
    ledger, workers = Ledger(), []
    original = r._write_new_json
    def fail_admission(path, payload):
        if Path(path).name == 'resource_admission.json':
            raise OSError('admission report filesystem failure')
        return original(path, payload)
    monkeypatch.setattr(r, '_write_new_json', fail_admission)
    def factory(n, directory):
        worker = Worker(ledger, n)
        workers.append(worker)
        return worker
    with pytest.raises(OSError):
        r.run_resource_probe(ledger=ledger, segment_id='summary_error', calibration={'value_scale': 2.},
            output_dir=tmp_path / 'summary_error', code_sha256='c' * 64,
            candidate_factory=factory, monitor_factory=lambda n: Monitor())
    assert len(workers) == 1 and workers[0].closed
    assert ledger.duties['probe_2048']['spent'] == 2048 * 57 * 48 + 101 * 5005
    assert not ledger.duties['probe_2048']['closed']
    assert all(not ledger.duties[f'probe_{n}'].get('skipped') for n in (1536, 1024, 512))
    assert (tmp_path / 'summary_error/n2048/result.json').exists()


def test_external_child_checks_exact_parent_reservation_without_settlement():
    row = dict(stage='S1', course='S1_RESOURCE', category='engineering', num_envs=100,
               max_control_steps=5005, work_item_id='extra_002', checkpoint_sha256='c' * 64,
               obligation_id='probe_2048', settled=False)
    snapshot = {'operations': {'key': row}}
    assert r.validate_child_reservation(snapshot, 'key', mode='evaluation', checkpoint_sha256='c' * 64) is row
    for change in ({'settled': True}, {'num_envs': 20}, {'max_control_steps': 5004},
                   {'work_item_id': 'extra_003'}, {'checkpoint_sha256': 'd' * 64}):
        with pytest.raises(ValueError):
            r.validate_child_reservation({'operations': {'key': dict(row, **change)}}, 'key',
                                          mode='evaluation', checkpoint_sha256='c' * 64)
    assert snapshot['operations']['key']['settled'] is False


def test_resource_child_cases_use_only_gate_A_copies_for_exact100_and1():
    from src.tasks.velocity.v09.cases import build_s1_manifest
    manifest = build_s1_manifest('gate_A')
    cases = r.resource_assessment_cases(manifest['cases'], num_envs=100)
    assert len(cases) == 100 and len({case['case_sha256'] for case in cases}) == 20
    assert all(case['suite'] == 'gate_A' and case['lesson'] == 'L40' for case in cases)
    assert len(r.resource_assessment_cases(manifest['cases'], num_envs=1)) == 1


def test_resource_gate20_is_replicated_five_times_without_changing_source_identity():
    from collections import Counter
    from src.tasks.velocity.v09.cases import build_s1_manifest
    manifest = build_s1_manifest('gate_A')
    original = {case['case_sha256']: case for case in manifest['cases']
                if case['lesson'] == 'L40' and case['speed'] == .3}
    expanded = r.resource_assessment_cases(manifest['cases'], num_envs=100)
    assert len(original) == 20 and len(expanded) == 100
    assert Counter(case['case_sha256'] for case in expanded) == {key: 5 for key in original}
    assert all(case == original[case['case_sha256']] for case in expanded)
    assert r.resource_assessment_cases(manifest['cases'], num_envs=1)[0] == expanded[0]


@pytest.mark.parametrize('corruption', ['missing', 'duplicate_sha', 'duplicate_identity', 'duplicate_case_key', 'timeout', 'gate_B', 'final'])
def test_resource_source_cases_reject_missing_duplicate_or_wrong_namespace(corruption):
    from copy import deepcopy
    from src.tasks.velocity.v09.cases import build_s1_manifest
    cases = deepcopy([case for case in build_s1_manifest('gate_A')['cases']
                      if case['lesson'] == 'L40' and case['speed'] == .3])
    if corruption == 'missing':
        cases.pop()
    elif corruption == 'duplicate_sha':
        cases[-1]['case_sha256'] = cases[0]['case_sha256']
    elif corruption == 'duplicate_identity':
        cases[-1]['case_id'] = cases[0]['case_id']
    elif corruption == 'duplicate_case_key':
        cases[-1]['case_key'] = cases[0]['case_key']
    elif corruption == 'timeout':
        cases[-1]['hard_timeout_steps'] = 5004
    else:
        for case in cases:
            case['suite'] = 'gate_B' if corruption == 'gate_B' else 'final_1'
    with pytest.raises(ValueError):
        r.resource_assessment_cases(cases, num_envs=100)


def test_resource_child_entrypoint_has_no_budget_reserve_or_commit_calls():
    source = (ROOT / 'scripts/a2_v09_resource_worker.py').read_text()
    assert '.reserve(' not in source and '.commit(' not in source
