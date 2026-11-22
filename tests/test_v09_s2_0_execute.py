"""CPU-only T5 execution orchestration tests; no GPU/physics."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v07_bootstrap

a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v09 import (s2_budget, s2_cases, s2_execute, s2_ledger)
import run_a2_v09_s2_0 as run_cli


@pytest.fixture(scope='module')
def plan():
    cases = ROOT / 'logs/a2_v09/s2_0/cases'
    matrix = s2_cases.load_frozen_manifest(cases / 'S2_DIAG.json', ROOT)
    h_series = s2_cases.load_frozen_manifest(cases / 'h_series_case0.json', ROOT)
    return s2_budget.build_reservation_plan(matrix, h_series)


class FakeRecorder:
    def __init__(self, case):
        self.case = case
        self.map_branch = 'real'
        self.rows = [dict(step=0), dict(step=1), dict(step=2)]


class FakeEnv:
    def __init__(self, cases, *, device, branch):
        self.cases = cases
        self.device = device
        self.branch = branch
        self.step_count = 0
        self.recorders = [FakeRecorder(case) for case in cases]

    def observation(self):
        return list(range(len(self.cases)))

    def step(self, actions):
        self.step_count += 1
        done = [self.step_count >= 3 for _ in self.cases]
        return self.observation(), done, {}

    def evidence(self, index):
        return dict(step=[0, 1, 2], map_evidence={'branch': self.branch},
                    state_summary={'state': 'SUCCESS'},
                    failure_context={'present': False},
                    g_level_report={'N': 2})

    def adapter(self, index):
        from src.tasks.velocity.v09 import s2_env
        return s2_env.S2TaskAdapter.from_case(self.cases[index])

    def recorder(self, index):
        return self.recorders[index]

    def metrics(self, index):
        return {'quality_pass': True}


def test_batch_table_has_30_frozen_batches():
    matrix = s2_cases.build_s2_0_manifest()
    run_plan = s2_cases.build_s2_diag_run_plan(matrix)
    batches = s2_execute.batch_table(run_plan)
    assert len(batches) == 30
    assert {row['batch_size'] for row in batches} == {5}
    assert {row['branch'] for row in batches} == {'real', 'plane_diagnostic'}
    assert sum(row['control_samples'] for row in batches) == 420_390


def test_reservation_gate_requires_stage_and_commits_full_padding(tmp_path, plan):
    ledger = s2_ledger.S2Ledger(tmp_path / 'ledger.jsonl', run_id='r')
    gate = s2_execute.ReservationGate(ledger)
    with pytest.raises(s2_execute.ReservationError):
        gate.require_stage()
    ledger.reserve_stage(plan)
    batch = dict(branch='real', group_role_sha256='g' * 64,
                 control_samples=10, case_sha256s=[], case_ids=[0])
    gate.begin_batch(batch, operation_id='batch')
    settled = gate.finish_batch(batch, operation_id='batch', actual_samples=7)
    assert settled['committed'] == 7
    assert settled['released'] == 3
    duty = ledger.state()['obligations']['s2_0_matrix_two_branches']
    assert duty['spent'] == 7
    assert duty['reserved'] == 0


def test_fake_driver_runs_batches_and_latches_terminal(tmp_path, plan):
    cases = s2_cases.build_s2_0_manifest()['cases'][:1]
    ledger = s2_ledger.S2Ledger(tmp_path / 'ledger.jsonl', run_id='r')
    ledger.reserve_stage(plan)
    driver = s2_execute.S2ExecutionDriver(
        env_factory=FakeEnv, policy=lambda obs: obs, ledger=ledger,
        output_dir=tmp_path / 'out', device='cpu', max_steps=10)
    batch = dict(branch='real', group_role_sha256='g' * 64,
                 control_samples=10, case_sha256s=[cases[0]['case_sha256']],
                 case_ids=[0], task=cases[0]['task'], speed=cases[0]['speed'],
                 platform_width=cases[0]['geometry'].get('platform_width'),
                 max_hard_timeout_steps=10, operation_ids=[])
    result = driver.run_batch(batch, cases, branch='real', operation_id='b0')
    assert result['control_steps'] == 3
    assert len(result['records']) == 1
    assert result['records'][0]['g_level_report']['N'] == 2
    assert result['settled']['committed'] == 3


def test_smoke_cli_rejects_without_ledger(tmp_path):
    with pytest.raises(SystemExit):
        run_cli.main(['--execute', '--smoke', '--case', '0',
                      '--ledger', str(tmp_path / 'missing.jsonl')])


def test_plan_only_cli_has_thirty_batches(capsys):
    code = run_cli.main(['--plan-only'])
    assert code == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc['batches'] == 30
    assert doc['physics_started'] is False


def test_attempt_ids_allow_retry_but_reject_same_operation(tmp_path, plan):
    ledger = s2_ledger.S2Ledger(tmp_path / 'ledger.jsonl', run_id='r')
    ledger.reserve_stage(plan)
    gate = s2_execute.ReservationGate(ledger)
    batch = dict(branch='real', group_role_sha256='g', control_samples=10,
                 case_sha256s=[], case_ids=[0])
    gate.begin_batch(batch, operation_id='smoke:0:attempt1')
    with pytest.raises(ValueError):
        gate.begin_batch(batch, operation_id='smoke:0:attempt1')
    gate.finish_batch(batch, operation_id='smoke:0:attempt1', actual_samples=0)
    gate.begin_batch(batch, operation_id='smoke:0:attempt2')
    gate.finish_batch(batch, operation_id='smoke:0:attempt2', actual_samples=0)
    state = ledger.state()
    assert state['obligations']['s2_0_matrix_two_branches']['spent'] == 0
    assert state['obligations']['s2_0_matrix_two_branches']['reserved'] == 0


def test_driver_records_attempt_id(tmp_path, plan):
    cases = s2_cases.build_s2_0_manifest()['cases'][:1]
    ledger = s2_ledger.S2Ledger(tmp_path / 'ledger.jsonl', run_id='r')
    ledger.reserve_stage(plan)
    driver = s2_execute.S2ExecutionDriver(
        env_factory=FakeEnv, policy=lambda obs: obs, ledger=ledger,
        output_dir=tmp_path / 'out', device='cpu', max_steps=10,
        attempt_id='attempt-xyz')
    batch = dict(branch='real', group_role_sha256='g', control_samples=10,
                 case_sha256s=[cases[0]['case_sha256']], case_ids=[0],
                 task=cases[0]['task'], speed=cases[0]['speed'],
                 platform_width=0.8, max_hard_timeout_steps=10, operation_ids=[])
    result = driver.run_batch(batch, cases, branch='real',
                              operation_id='b0:attempt-xyz')
    assert result['records'][0]['attempt_id'] == 'attempt-xyz'
