"""Manual launcher wiring, without importing a simulator or running physics."""
import importlib.util
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]


def launcher(monkeypatch):
    calls = []
    class Controller:
        def __init__(self, *args, **kwargs):
            calls.append(('init', args, kwargs))
        def run(self, **kwargs):
            calls.append(('run', kwargs))
            return {'passed': True}
    fake = types.ModuleType('src.tasks.velocity.v10.orchestration')
    fake.Orchestrator = Controller
    fake.preflight_formal_handoff = lambda *a: calls.append(('check', a)) or {'passed': True}
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    spec = importlib.util.spec_from_file_location('manual_launcher', ROOT/'scripts/run_a2_v10_formal.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, calls


def test_check_does_not_construct_controller_or_run(monkeypatch):
    module, calls = launcher(monkeypatch)
    assert module.main(['--mode','check','--device','cuda:0','--num-envs','1024','--rollout','48','--seed','42']) == 0
    assert [c[0] for c in calls] == ['check']


def test_all_runs_full_flow_with_original_shared_ledgers(monkeypatch, tmp_path):
    module, calls = launcher(monkeypatch)
    module.ROOT = tmp_path
    run = tmp_path/'logs/a2_v10/formal_A1_20261127_approved'
    selection = run/'resource_probe_v2_sds_resume_20261127_183400/selection.json'
    selection.parent.mkdir(parents=True)
    selection.write_text('{"probe_sequence_id":"existing-sequence"}')
    assert module.main(['--mode','all']) == 0
    assert calls[-1] == ('run', {'teachers_only':False})
    kwargs = calls[-2][2]
    assert kwargs['learning_ledger'] == run/'learning_costs.jsonl'
    assert kwargs['evaluation_ledger'] == tmp_path/'logs/a2_v10/formal_A_20261127_approved/evaluation_costs.jsonl'
    assert kwargs['probe_sequence_id'] == 'existing-sequence'
    assert kwargs['formal_handoff_plan'].name == 'formal_handoff_plan_20261127.json'


def test_recipe_changes_rejected_before_preflight(monkeypatch):
    module, calls = launcher(monkeypatch)
    with pytest.raises(SystemExit):
        module.main(['--num-envs','512'])
    assert not calls
