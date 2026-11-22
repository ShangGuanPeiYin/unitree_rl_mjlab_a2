"""CPU-only tests for frozen S2.1 A/B gate case manifests."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v07_bootstrap

a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v09 import s2_gate_cases


def test_gate_manifest_shape_budget_and_l40_cache():
    a = s2_gate_cases.build_gate_manifest('gate_A')
    b = s2_gate_cases.build_gate_manifest('gate_B')
    for manifest in (a, b):
        s2_gate_cases.validate_gate_manifest(manifest)
        assert manifest['group_count'] == 36
        assert len(manifest['groups']) == 36
        assert len(manifest['cases']) == 720
        assert manifest['budget']['control_samples'] == 1_620_000
        assert manifest['budget']['l40_milestone_groups'] == 3
        assert manifest['budget']['l40_milestone_samples'] == 219_220
        assert all(group['cases'] == 20 for group in manifest['groups'])
        assert sum(group['task'] == 'legacy_l40' for group in manifest['groups']) == 3
    assert a['suite'] == 'S2_GATE_A'
    assert b['suite'] == 'S2_GATE_B'
    assert a['manifest_sha256'] != b['manifest_sha256']
    assert {row['case_sha256'] for row in a['cases']}.isdisjoint(
        {row['case_sha256'] for row in b['cases']})


def test_gate_manifest_is_deterministic_and_tamper_evident():
    first = s2_gate_cases.build_gate_manifest('gate_A')
    second = s2_gate_cases.build_gate_manifest('gate_A')
    assert first['manifest_sha256'] == second['manifest_sha256']
    assert first['cases'] == second['cases']
    changed = json.loads(json.dumps(first))
    changed['cases'][0]['speed'] += 0.01
    changed['manifest_sha256'] = s2_gate_cases.sha256(
        {key: value for key, value in changed.items() if key != 'manifest_sha256'})
    with pytest.raises(ValueError, match='case_key|case SHA'):
        s2_gate_cases.validate_gate_manifest(changed)


def test_gate_case_keys_and_independent_suite_streams():
    a = s2_gate_cases.build_gate_manifest('gate_A')
    case = a['cases'][0]
    assert case['case_key'] == [
        'S2', 'S2_GATE_A', case['task'], case['stair_count'], case['geometry'],
        case['speed'], case['stop_mode'], case['force_family'], case['case_id']]
    assert case['streams'] != s2_gate_cases.build_gate_manifest('gate_B')[
        'cases'][0]['streams']


def test_freeze_is_append_only(tmp_path):
    receipt = s2_gate_cases.freeze_gate_cases(tmp_path)
    assert set(receipt) == {'schema_version', 'gate_version', 'course', 'a', 'b'}
    assert Path(receipt['a']['path']).name == 'S2_GATE_A.json'
    assert Path(receipt['b']['path']).name == 'S2_GATE_B.json'
    assert json.loads(Path(receipt['a']['path']).read_text())['frozen'] is True
    with pytest.raises(FileExistsError):
        s2_gate_cases.freeze_gate_cases(tmp_path)
