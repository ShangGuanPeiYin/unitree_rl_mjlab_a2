"""Q009 disposable parent/child boundaries; CPU models and synthetic states only."""
from copy import deepcopy
import hashlib
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from a2_v07_bootstrap import activate
activate(ROOT)
from src.tasks.velocity.v09 import resources as r
from src.tasks.velocity.v09.env import EpisodeIds
from src.tasks.velocity.v09.model import V09ActorCritic, Frozen6400Teacher, synthetic_observations


class TrainingEnv:
    def __init__(self):
        self.closed = False
    def close(self):
        self.closed = True


@pytest.fixture
def parent(tmp_path):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    worker = object.__new__(r.RealProbeCandidate)
    worker.num_envs, worker.device, worker.course = 512, 'cpu', 'S1.2'
    worker.identity_context = None
    worker.output_dir, worker.code_sha256, worker.ledger_run_id = tmp_path, 'b' * 64, 'test'
    worker.calibration = {'value_scale': 2., 'manifest_sha256': 'a' * 64}
    worker.student = V09ActorCritic.migrate_6400_checkpoint()
    worker.student.set_s1_course('S1.2')
    worker.teacher = Frozen6400Teacher.from_6400_checkpoint()
    worker.optimizer = torch.optim.Adam(worker.student.optimizer_parameters(), lr=3e-5)
    obs = synthetic_observations(batch=2)
    (worker.student(obs).square().mean() + worker.student.evaluate(obs).square().mean()).backward()
    worker.optimizer.step()
    worker.generator = torch.Generator().manual_seed(42)
    worker.roles = ['flat_walk'] * 512
    worker.ids = EpisodeIds({'flat_walk': 512})
    worker.env, worker.eval_env, worker.buffers = TrainingEnv(), None, [{'full': True}, {'full': True}]
    worker.iterations, worker.frozen = 56, worker._frozen_digest()
    def new_env():
        worker.ids.next_by_role['flat_walk'] += 512
        return TrainingEnv()
    worker._new_env = new_env
    yield worker
    torch.set_num_threads(old_threads)


def test_cpu_offload_preserves_parameter_and_adam_mapping_restores_rng_new_role_ids(parent):
    identities = parent._parameter_identity()
    old_env = parent.env
    info = parent.offload_for_evidence()
    assert old_env.closed and parent.env is None and parent.buffers == [None, None]
    assert parent._parameter_identity() == identities
    random.random()
    np.random.random()
    torch.rand(2)
    restored = parent.restore_after_evidence()
    assert restored['checkpoint_sha256'] == info['checkpoint_sha256']
    assert restored['fingerprints_verified'] and restored['rng_restored']
    assert restored['parameter_optimizer_identity_preserved']
    assert parent._parameter_identity() == identities
    assert parent.ids.state_dict()['flat_walk'] == 1024
    assert all(value.device.type == 'cpu' for state in parent.optimizer.state.values()
               for value in state.values() if torch.is_tensor(value))


def test_offloaded_parent_parameter_corruption_fails_before_rebuilding_env(parent):
    parent.offload_for_evidence()
    with torch.no_grad():
        next(parent.student.parameters()).add_(1)
    with pytest.raises(RuntimeError, match='identity changed'):
        parent.restore_after_evidence()
    assert parent.env is None


def test_parent_restore_rejects_legacy_producer_even_when_expected_file_hash_matches(parent):
    info = parent.offload_for_evidence()
    path = Path(info['path'])
    payload = torch.load(path, map_location='cpu', weights_only=False)
    payload['metadata']['code_sha256'] = '44780cd5d6f9ee4029fb38b2f9348a872268e7bedf995d4b36e2e30ec2f2163f'
    torch.save(payload, path)
    # A matching content digest alone cannot authorize a different producer.
    parent.external_checkpoint['checkpoint_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='code SHA'):
        parent.restore_after_evidence()
    assert parent.env is None


def test_external_evaluation_advances_all5005_intervals_after_first_terminal():
    class AssessmentEnv:
        num_envs = 100
        steps = 0
        def observation_pair(self):
            return {}, {}
        def step(self, action):
            self.steps += 1
            rows = [dict(env_id=i, terminal_step=1, failure_masks={}) for i in range(100)] if self.steps == 1 else []
            return {}, None, None, None, {'episodes': rows}
    env = AssessmentEnv()
    result = r.collect_padded_assessment(env, lambda obs: None, control_steps=5005)
    assert env.steps == result['control_steps'] == 5005
    assert result['physical_control_samples'] == 100 * 5005
    assert all(row['terminal_step'] == 1 and row['operation_course'] == 'S1_RESOURCE'
               for row in result['first_terminal_records'])


def test_reserved_external_child_result_is_bound_to_operation_and_checkpoint(tmp_path, monkeypatch):
    worker = object.__new__(r.RealProbeCandidate)
    worker.env, worker.buffers, worker.output_dir = None, [None, None], tmp_path
    worker.device, worker.code_sha256, worker.ledger_run_id = 'cpu', 'b' * 64, 'run'
    worker.run_dir = tmp_path
    worker.identity_context = None
    worker.ledger_path, worker.gate_manifest_path = tmp_path / 'ledger', tmp_path / 'gate_A.json'
    worker.external_checkpoint = {'path': str(tmp_path / 'parent.pt'), 'checkpoint_sha256': 'a' * 64}
    def subprocess_mock(command, **kwargs):
        assert command[command.index('--run-dir') + 1] == str(tmp_path)
        assert command[command.index('--run-id') + 1] == 'run'
        directory = tmp_path / 'external_evaluation'
        directory.mkdir()
        data = dict(checkpoint_sha256='a' * 64, operation_key='reserved-key',
                    sds_sha256=r.spec.SDS_SHA256, code_sha256='b' * 64,
                    reservation_verified=True, num_envs=100, control_steps=5005)
        data['report_sha256'] = r.sha256(data)
        r._write_new_json(directory / 'worker_result.json', data)
        return type('Process', (), {'returncode': 0, 'stdout': '', 'stderr': ''})()
    monkeypatch.setattr(r.subprocess, 'run', subprocess_mock)
    result = worker.external_assessment('evaluation', 'reserved-key')
    assert result['num_envs'] == 100 and result['operation_key'] == 'reserved-key'


def test_parent_passes_verified_context_identity_to_external_worker(tmp_path, monkeypatch):
    from v09_identity_fixtures import make_identity_fixture
    from src.tasks.velocity.v09.evidence_identity import EvidenceIdentityContext
    from src.tasks.velocity.v09.cases import build_s1_manifest
    fixture = make_identity_fixture(tmp_path, consumer_code_sha256='b' * 64)
    gate = build_s1_manifest('gate_A')
    gate_path = fixture.put_json(fixture.run_dir / 'cases/gate_A.json', gate)
    fixture.bind_file(gate_path, kind='frozen_manifest', suite='gate_A',
                      manifest_sha256=gate['manifest_sha256'], category='frozen_inputs')
    fixture.seal()
    context = EvidenceIdentityContext.from_approval(fixture.approval_path, fixture.approval_sha256,
                                                    **fixture.context_kwargs())
    worker = object.__new__(r.RealProbeCandidate)
    worker.env, worker.buffers, worker.output_dir = None, [None, None], tmp_path
    worker.device, worker.code_sha256, worker.ledger_run_id = 'cpu', 'b' * 64, fixture.run_id
    worker.run_dir, worker.identity_context = fixture.run_dir, context
    worker.ledger_path, worker.gate_manifest_path = fixture.ledger_path, gate_path
    worker.external_checkpoint = {'path': str(tmp_path / 'parent.pt'), 'checkpoint_sha256': 'a' * 64}
    def subprocess_mock(command, **kwargs):
        assert command[command.index('--evidence-identity') + 1] == str(fixture.approval_path.resolve())
        assert command[command.index('--evidence-identity-sha256') + 1] == fixture.approval_sha256
        assert command[command.index('--run-dir') + 1] == str(fixture.run_dir.resolve())
        assert command[command.index('--run-id') + 1] == fixture.run_id
        directory = tmp_path / 'external_evaluation'
        directory.mkdir()
        data = dict(checkpoint_sha256='a' * 64, operation_key='reserved-key',
                    sds_sha256=r.spec.SDS_SHA256, code_sha256='b' * 64,
                    reservation_verified=True, num_envs=100, control_steps=5005,
                    manifest_evidence_ref={'approval_sha256': fixture.approval_sha256})
        data['report_sha256'] = r.sha256(data)
        r._write_new_json(directory / 'worker_result.json', data)
        return type('Process', (), {'returncode': 0, 'stdout': '', 'stderr': ''})()
    monkeypatch.setattr(r.subprocess, 'run', subprocess_mock)
    result = worker.external_assessment('evaluation', 'reserved-key')
    assert result['manifest_evidence_ref']['approval_sha256'] == fixture.approval_sha256
