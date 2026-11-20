"""Real resource CLI preflight regressions; CUDA disabled, no physics or ledger changes."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from a2_v07_bootstrap import activate
activate(ROOT)
from src.tasks.velocity.v09 import spec
from src.tasks.velocity.v09.budget import AppendOnlyLedger
from src.tasks.velocity.v09.cases import build_s1_manifest, freeze_manifest
from src.tasks.velocity.v09.cases import sha256


def run_cli(script, arguments):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    return subprocess.run([sys.executable, str(ROOT / 'scripts' / script), *map(str, arguments)],
                          cwd=ROOT, env=environment, capture_output=True, text=True, timeout=60)


def probe_inputs(tmp_path):
    run_dir = tmp_path / 'run'
    (run_dir / 'cases').mkdir(parents=True)
    gate = build_s1_manifest('gate_A')
    train = build_s1_manifest('train_calibration', calibration_cases_per_group=1)
    freeze_manifest(run_dir / 'cases/gate_A.json', gate)
    freeze_manifest(run_dir / 'cases/train_calibration.json', train)
    ledger_path = run_dir / 'budget.jsonl'
    AppendOnlyLedger(ledger_path, run_id='resource-cli-test')
    code_sha = spec.code_snapshot()['code_sha256']
    calibration = dict(sds_sha256=spec.SDS_SHA256, source_sha256=spec.SOURCE_SHA256,
                       manifest_sha256=train['manifest_sha256'], code_sha256=code_sha,
                       value_scale=2., value_samples=300, physical_control_samples=400,
                       method='population_std_max_1')
    calibration_path = run_dir / 'calibration.json'
    calibration_path.write_text(json.dumps(calibration))
    output = tmp_path / 'must-not-be-created'
    args = ['--run-dir', run_dir, '--run-id', 'resource-cli-test', '--segment-id', 'dry-run',
            '--code-sha256', code_sha, '--calibration', calibration_path, '--output-dir', output,
            '--device', 'cuda:0', '--physical-gpu-index', '0', '--monitor-interval', '0.02']
    return args, ledger_path, output


def test_real_probe_cli_accepts_twenty_sources_as_one_hundred_slots_without_side_effects(tmp_path):
    args, ledger, output = probe_inputs(tmp_path)
    before = ledger.read_bytes()
    result = run_cli('probe_a2_v09_s1.py', args)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['source_case_count'] == 20
    assert report['evaluation_num_envs'] == 100
    assert report['evaluation_control_steps'] == 5005
    assert report['formal_physics_started'] is False
    assert not output.exists()
    assert ledger.read_bytes() == before


def worker_inputs(tmp_path, *, mode='evaluation', corruption=None, identity_fixture=None):
    import numpy as np
    import torch
    from src.tasks.velocity.v09.checkpoint import save_checkpoint
    from src.tasks.velocity.v09.model import V09ActorCritic
    run_dir = identity_fixture.run_dir if identity_fixture is not None else tmp_path / 'worker-run'
    run_id = identity_fixture.run_id if identity_fixture is not None else 'worker-cli-test'
    (run_dir / 'cases').mkdir(parents=True, exist_ok=True)
    gate = build_s1_manifest('gate_A')
    freeze_manifest(run_dir / 'cases/gate_A.json', gate)
    code_sha = identity_fixture.consumer_code_sha256 if identity_fixture is not None else spec.code_snapshot()['code_sha256']
    metadata = dict(stage='S1', course='S1.1' if corruption == 'course' else 'S1.2',
        value_scale=2., manifest_sha256='a' * 64, ledger_run_id=run_id, learning_samples=0,
        code_sha256=('44780cd5d6f9ee4029fb38b2f9348a872268e7bedf995d4b36e2e30ec2f2163f'
                     if corruption == 'old_producer' else code_sha),
        gate_state=dict(next_suite='gate_A', attempts=0, final_suites_used=0,
                        recoveries=0, next_gate_samples=0, passes=[]),
        role_state=dict(num_envs=512, episode_counters={'flat_walk': 512}, assignments=['flat_walk'] * 512),
        random_state=dict(python=None, numpy=np.random.get_state()))
    model = V09ActorCritic.migrate_6400_checkpoint()
    optimizer = torch.optim.Adam(model.optimizer_parameters(), lr=3e-5)
    checkpoint = run_dir / 'parent.pt'
    save_checkpoint(checkpoint, model, optimizer, metadata)
    if corruption == 'schema':
        payload = torch.load(checkpoint, weights_only=False, map_location='cpu')
        payload['schema'] = 10
        torch.save(payload, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    ledger_path = run_dir / 'budget.jsonl'
    ledger = identity_fixture.ledger if identity_fixture is not None else AppendOnlyLedger(ledger_path, run_id=run_id)
    work = {'extra_002': dict(num_envs=100, control_steps=5005),
            'extra_003': dict(num_envs=1, control_steps=5005)}
    plan = dict(stage='S1', admitted=True, sds_sha256=spec.SDS_SHA256,
        total_control_samples=505505, case_catalogue={}, obligations={
            'probe_512': dict(kind='probe', control_samples=505505, work_items=work),
            'engineering': dict(kind='engineering', control_samples=0)})
    plan['plan_sha256'] = sha256(plan)
    ledger.reserve_stage(plan)
    n, item = (100, 'extra_002') if mode == 'evaluation' else (1, 'extra_003')
    operation = ledger.reserve(segment_id='worker-cli', operation_id=mode, stage='S1',
        course='S1_RESOURCE', category='engineering', num_envs=n, max_control_steps=5005,
        checkpoint_sha256=digest, obligation_id='probe_512', work_item_id=item, reason='CLI regression fixture')
    output = tmp_path / 'child-output-must-not-exist'
    args = ['--mode', mode, '--run-dir', run_dir, '--ledger', ledger_path, '--run-id', run_id, '--operation-key', operation,
            '--checkpoint', checkpoint, '--checkpoint-sha256', digest, '--code-sha256', code_sha,
            '--gate-manifest', run_dir / 'cases/gate_A.json', '--device', 'cuda:0', '--num-envs', str(n),
            '--control-steps', '5005', '--width', '640', '--height', '480', '--fps', '25', '--output-dir', output]
    if identity_fixture is not None:
        identity_fixture.bind_file(run_dir / 'cases/gate_A.json', kind='frozen_manifest', suite='gate_A',
                                   manifest_sha256=gate['manifest_sha256'], category='frozen_inputs')
        identity_fixture.seal()
        args += ['--evidence-identity', identity_fixture.approval_path,
                 '--evidence-identity-sha256', identity_fixture.approval_sha256]
    return args, ledger_path, output


@pytest.mark.parametrize('mode', ['evaluation', 'video'])
def test_real_child_cli_preflights_new_checkpoint_without_physics_or_ledger_write(tmp_path, mode):
    args, ledger, output = worker_inputs(tmp_path, mode=mode)
    before = ledger.read_bytes()
    result = run_cli('a2_v09_resource_worker.py', args)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['reservation_verified'] is True and report['physical_control_steps'] == 0
    assert report['num_envs'] == (100 if mode == 'evaluation' else 1)
    assert report['control_steps'] == 5005
    assert not output.exists() and ledger.read_bytes() == before


def identity_probe_inputs(tmp_path):
    from v09_identity_fixtures import make_identity_fixture
    fixture = make_identity_fixture(tmp_path, consumer_code_sha256=spec.code_snapshot()['code_sha256'])
    gate = build_s1_manifest('gate_A')
    train = build_s1_manifest('train_calibration', calibration_cases_per_group=8)
    for suite, manifest in (('gate_A', gate), ('train_calibration', train)):
        path = fixture.put_json(fixture.run_dir / 'cases' / f'{suite}.json', manifest)
        fixture.bind_file(path, kind='frozen_manifest', suite=suite,
                          manifest_sha256=manifest['manifest_sha256'], category='frozen_inputs')
    fixture.calibration['manifest_sha256'] = train['manifest_sha256']
    fixture.put_json(fixture.calibration_path, fixture.calibration)
    fixture.bind_file(fixture.calibration_path, kind='calibration', suite='train_calibration',
                      manifest_sha256=train['manifest_sha256'])
    fixture.seal()
    output = tmp_path / 'legacy-probe-must-not-exist'
    args = ['--run-dir', fixture.run_dir, '--run-id', fixture.run_id, '--segment-id', 'legacy-preflight',
            '--code-sha256', fixture.consumer_code_sha256, '--calibration', fixture.calibration_path,
            '--evidence-identity', fixture.approval_path, '--evidence-identity-sha256', fixture.approval_sha256,
            '--output-dir', output, '--device', 'cuda:0', '--physical-gpu-index', '0', '--monitor-interval', '0.02']
    return fixture, args, output


def test_real_probe_cli_reads_approved_legacy_inputs_without_rewriting_producer(tmp_path):
    fixture, args, output = identity_probe_inputs(tmp_path)
    ledger_before, calibration_before = fixture.ledger_path.read_bytes(), fixture.calibration_path.read_bytes()
    result = run_cli('probe_a2_v09_s1.py', args)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['source_case_count'] == 20 and report['evaluation_num_envs'] == 100
    assert report['calibration']['code_sha256'] == fixture.calibration['code_sha256']
    assert report['calibration']['evidence_ref']['approval_sha256'] == fixture.approval_sha256
    assert report['code_sha256'] == fixture.consumer_code_sha256
    assert not output.exists()
    assert fixture.ledger_path.read_bytes() == ledger_before
    assert fixture.calibration_path.read_bytes() == calibration_before


@pytest.mark.parametrize('corruption', ['approval_sha', 'target', 'manifest_bytes', 'missing_sha'])
def test_real_probe_cli_rejects_invalid_legacy_identity_before_side_effects(tmp_path, corruption):
    fixture, args, output = identity_probe_inputs(tmp_path)
    if corruption == 'approval_sha':
        args[args.index('--evidence-identity-sha256') + 1] = '0' * 64
    elif corruption == 'target':
        fixture.approval['consumer_code_sha256'] = 'd' * 64
        fixture.seal()
        args[args.index('--evidence-identity-sha256') + 1] = fixture.approval_sha256
    elif corruption == 'manifest_bytes':
        gate_path = fixture.run_dir / 'cases/gate_A.json'
        gate_path.write_text(gate_path.read_text() + ' ')
    else:
        position = args.index('--evidence-identity-sha256')
        del args[position:position + 2]
    ledger_before = fixture.ledger_path.read_bytes()
    result = run_cli('probe_a2_v09_s1.py', args)
    assert result.returncode != 0
    assert not output.exists() and fixture.ledger_path.read_bytes() == ledger_before


@pytest.mark.parametrize('corruption', ['approval_sha', 'target'])
def test_real_resource_child_rejects_bad_context_even_if_parent_supplies_it(tmp_path, corruption):
    from v09_identity_fixtures import make_identity_fixture
    fixture = make_identity_fixture(tmp_path, consumer_code_sha256=spec.code_snapshot()['code_sha256'])
    args, ledger, output = worker_inputs(tmp_path, identity_fixture=fixture)
    if corruption == 'approval_sha':
        args[args.index('--evidence-identity-sha256') + 1] = '0' * 64
    else:
        fixture.approval['consumer_code_sha256'] = 'd' * 64
        fixture.seal()
        args[args.index('--evidence-identity-sha256') + 1] = fixture.approval_sha256
    before = ledger.read_bytes()
    result = run_cli('a2_v09_resource_worker.py', args)
    assert result.returncode != 0
    assert 'approval' in result.stderr or 'consumer' in result.stderr
    assert not output.exists() and ledger.read_bytes() == before


def test_legacy_approval_does_not_waive_actual_parent_source_snapshot(tmp_path):
    fixture, args, output = identity_probe_inputs(tmp_path)
    args[args.index('--code-sha256') + 1] = fixture.calibration['code_sha256']
    before = fixture.ledger_path.read_bytes()
    result = run_cli('probe_a2_v09_s1.py', args)
    assert result.returncode != 0 and 'source snapshot' in result.stderr
    assert not output.exists() and fixture.ledger_path.read_bytes() == before


@pytest.mark.parametrize('corruption', ['missing', 'duplicate', 'timeout', 'gate_B', 'final'])
def test_real_probe_cli_rejects_invalid_source_manifest_before_side_effects(tmp_path, corruption):
    args, ledger, output = probe_inputs(tmp_path)
    gate_path = Path(args[args.index('--run-dir') + 1]) / 'cases/gate_A.json'
    if corruption in ('gate_B', 'final'):
        gate = build_s1_manifest('gate_B' if corruption == 'gate_B' else 'final_1')
    else:
        gate = json.loads(gate_path.read_text())
        indices = [i for i, case in enumerate(gate['cases']) if case['lesson'] == 'L40' and case['speed'] == .3]
        if corruption == 'missing':
            del gate['cases'][indices[-1]]
        elif corruption == 'duplicate':
            gate['cases'][indices[-1]] = gate['cases'][indices[0]]
        else:
            gate['cases'][indices[-1]]['hard_timeout_steps'] = 5004
        gate['manifest_sha256'] = sha256({key: value for key, value in gate.items() if key != 'manifest_sha256'})
    gate_path.write_text(json.dumps(gate))
    before = ledger.read_bytes()
    result = run_cli('probe_a2_v09_s1.py', args)
    assert result.returncode != 0
    assert not output.exists() and ledger.read_bytes() == before


@pytest.mark.parametrize('mode', ['evaluation', 'video'])
def test_real_resource_child_rebuilds_legacy_gate_context_independently(tmp_path, mode):
    from v09_identity_fixtures import make_identity_fixture
    fixture = make_identity_fixture(tmp_path, consumer_code_sha256=spec.code_snapshot()['code_sha256'])
    args, ledger, output = worker_inputs(tmp_path, mode=mode, identity_fixture=fixture)
    before = ledger.read_bytes()
    result = run_cli('a2_v09_resource_worker.py', args)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['reservation_verified'] and report['physical_control_steps'] == 0
    assert report['manifest_evidence_ref']['approval_sha256'] == fixture.approval_sha256
    assert not output.exists() and ledger.read_bytes() == before


@pytest.mark.parametrize('mode', ['evaluation', 'video'])
@pytest.mark.parametrize('alter_manifest_bytes', [False, True])
def test_fixed_legacy_run_child_requires_both_identity_flags_even_for_json_equivalent_bytes(
        tmp_path, mode, alter_manifest_bytes):
    from v09_identity_fixtures import make_identity_fixture
    fixture = make_identity_fixture(tmp_path, consumer_code_sha256=spec.code_snapshot()['code_sha256'])
    args, ledger, output = worker_inputs(tmp_path, mode=mode, identity_fixture=fixture)
    for flag in ('--evidence-identity', '--evidence-identity-sha256'):
        position = args.index(flag)
        del args[position:position + 2]
    if alter_manifest_bytes:
        gate_path = Path(args[args.index('--gate-manifest') + 1])
        gate_path.write_bytes(gate_path.read_bytes() + b' ')
    before = ledger.read_bytes()
    result = run_cli('a2_v09_resource_worker.py', args)
    assert result.returncode != 0, 'fixed r3 child bypassed its exact-byte identity catalog'
    assert 'identity' in result.stderr.lower()
    assert result.stdout == ''
    assert not output.exists() and ledger.read_bytes() == before


@pytest.mark.parametrize('mode', ['evaluation', 'video'])
@pytest.mark.parametrize('missing_flag', ['--evidence-identity', '--evidence-identity-sha256'])
def test_fixed_legacy_run_child_rejects_either_missing_identity_flag(tmp_path, mode, missing_flag):
    from v09_identity_fixtures import make_identity_fixture
    fixture = make_identity_fixture(tmp_path, consumer_code_sha256=spec.code_snapshot()['code_sha256'])
    args, ledger, output = worker_inputs(tmp_path, mode=mode, identity_fixture=fixture)
    position = args.index(missing_flag)
    del args[position:position + 2]
    before = ledger.read_bytes()
    result = run_cli('a2_v09_resource_worker.py', args)
    assert result.returncode != 0
    assert 'both evidence identity' in result.stderr
    assert result.stdout == ''
    assert not output.exists() and ledger.read_bytes() == before


def test_real_resource_child_legacy_context_never_authorizes_old_student_checkpoint(tmp_path):
    from v09_identity_fixtures import make_identity_fixture
    fixture = make_identity_fixture(tmp_path, consumer_code_sha256=spec.code_snapshot()['code_sha256'])
    args, ledger, output = worker_inputs(tmp_path, corruption='old_producer', identity_fixture=fixture)
    before = ledger.read_bytes()
    result = run_cli('a2_v09_resource_worker.py', args)
    assert result.returncode != 0 and 'code SHA' in result.stderr
    assert not output.exists() and ledger.read_bytes() == before


@pytest.mark.parametrize('corruption,diagnostic', [
    ('old_producer', 'full S1.2 parent'), ('course', 'full S1.2 parent'), ('schema', 'checkpoint schema mismatch')])
def test_real_child_cli_rejects_old_or_wrong_checkpoint_before_execute(tmp_path, corruption, diagnostic):
    args, ledger, output = worker_inputs(tmp_path, corruption=corruption)
    before = ledger.read_bytes()
    result = run_cli('a2_v09_resource_worker.py', args)
    assert result.returncode != 0, 'dry-run accepted an incompatible checkpoint'
    assert diagnostic in result.stderr or (corruption == 'old_producer' and 'code SHA' in result.stderr)
    assert not output.exists() and ledger.read_bytes() == before
