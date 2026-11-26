"""v09 play CLI must avoid the generic task scanner and preserve frozen cases."""
import importlib.util
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def load_player():
    spec = importlib.util.spec_from_file_location('v09_player', ROOT / 'scripts/play_a2_v09_s1.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_play_route_help_is_isolated():
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/play.py'),
                             'Unitree-A2-V09-S1', '--help'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--lesson' in result.stdout
    assert '--checkpoint-sha256' in result.stdout
    assert 'ObservationManager' not in result.stdout


def test_frozen_case_selection_and_rejection():
    import pytest
    player = load_player()
    manifest = {'cases': [dict(task='stairs', lesson='L40', speed=0.5, case_id=0,
                               command=dict(vx=0.5, vy=0.0, wz=0.0))]}
    case = player.select_case(manifest, 'L40', 0.5, 0)
    assert case['lesson'] == 'L40' and case['speed'] == 0.5
    assert case['command']['vy'] == 0 and case['command']['wz'] == 0
    with pytest.raises(ValueError, match='没有唯一匹配'):
        player.select_case(manifest, 'L40', 0.123, 0)


def test_scan_extent_uses_cell_edges():
    import numpy as np
    player = load_player()
    x, y = np.meshgrid(-0.8 + (np.arange(40) + 0.5) * 0.05,
                       -0.75 + (np.arange(30) + 0.5) * 0.05, indexing='ij')
    assert np.allclose(player.scan_bounds(np.stack([x, y], -1), 0.05),
                       [-0.8, 1.2, -0.75, 0.75])
