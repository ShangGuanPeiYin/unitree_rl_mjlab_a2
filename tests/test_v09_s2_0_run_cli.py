"""CPU-only T10 case-selection tests; no ledger/GPU/physics."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v07_bootstrap

a2_v07_bootstrap.activate(ROOT)
import run_a2_v09_s2_0 as run_cli


@pytest.fixture(scope='module')
def inputs():
    return run_cli.load_inputs()


def test_case_id_alone_is_ambiguous_not_list_position(inputs):
    matrix, h_series, _ = inputs
    with pytest.raises(ValueError, match='not globally unique'):
        run_cli.select_case(matrix, h_series, suite='S2_DIAG', case_id=0)


def test_t10_staircase_identity_via_documented_filters(inputs):
    matrix, h_series, _ = inputs
    case, meta = run_cli.select_case(
        matrix, h_series, suite='S2_DIAG', case_id=0,
        task='stairs_walk', speed=0.3, platform_width=0.8)
    assert case['case_sha256'].startswith('0140473c4282')
    assert meta['case_id'] == 0
    assert meta['suite'] == 'S2_DIAG'
    assert meta['source'] == 'S2_DIAG.json'
    assert meta['task'] == 'stairs_walk'
    assert meta['stair_count'] == 40
    assert meta['speed'] == pytest.approx(0.3)
    assert meta['hard_timeout_steps'] == 6688
    assert meta['platform_front_local_x'] == pytest.approx(8.0)


def test_sha_prefix_selects_the_same_frozen_case(inputs):
    matrix, h_series, _ = inputs
    case, meta = run_cli.select_case(
        matrix, h_series, suite='S2_DIAG',
        case_sha256='0140473c4282')
    assert case['task'] == 'stairs_walk'
    assert case['hard_timeout_steps'] == 6688
    assert meta['case_sha256'].startswith('0140473c4282')


def test_h_series_requires_sha_because_ids_repeat(inputs):
    matrix, h_series, _ = inputs
    with pytest.raises(ValueError, match='not globally unique'):
        run_cli.select_case(matrix, h_series, suite='H_SERIES', case_id=0)
    case, meta = run_cli.select_case(
        matrix, h_series, suite='H_SERIES', case_sha256='84feecc2d400')
    assert case['task'] == 'stairs'
    assert case['hard_timeout_steps'] == 1301
    assert meta['suite'] == 'H_SERIES'
    assert meta['manifest_suite'] == 'S1_DIAG'
