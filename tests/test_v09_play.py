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


def test_temporary_geometry_keeps_frozen_training_spec():
    code = '''
import sys
sys.path.insert(0, 'scripts')
from a2_v07_bootstrap import activate
activate()
from play_a2_v09_s1 import configure_terrain
from src.tasks.velocity.v07 import terrain, spec
original = spec.terrain_spec(spec.tile_id_for('L40'))
case = {'geometry': {'tile_id': original.tile_id}}
configure_terrain(case, steps=20, width=2.0)
played = terrain.build_terrain_specs()[original.tile_id]
assert played.N == 20
assert terrain.platform_width(played) == 2.0
assert played.platform_length == original.platform_length == 2.0
assert played.h == original.h == 0.2
assert played.step_depth == original.step_depth == 0.2
assert played.width == original.width == 0.8
assert original.N == 40
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_s2_playback_requires_explicit_diagnostic_opt_in():
    import pytest
    player = load_player()
    payload = dict(schema=11, metadata=dict(stage='S2', course='S2.1'))
    with pytest.raises(ValueError, match='allow-s2-diagnostic'):
        player.playback_course(payload, allow_s2=False)
    assert player.playback_course(payload, allow_s2=True) == 'S1.2'
    assert player.playback_course(dict(schema=11, metadata=dict(stage='S1', course='S1.1')),
                                  allow_s2=False) == 'S1.1'


def test_continue_after_success_retains_failures_and_pre_top_timeout():
    import torch
    from dataclasses import dataclass
    player = load_player()

    @dataclass(frozen=True)
    class State:
        success: object
        failure: object
        timeout: object
        reason: object

    state = State(torch.tensor([True, False, False]), torch.tensor([False, True, False]),
                  torch.tensor([False, False, True]), torch.tensor([2, 3, 1]))
    result, reached = player.continuation_state(state, torch.zeros(3, dtype=torch.bool))
    assert reached.tolist() == [True, False, False]
    assert result.success.tolist() == [False, False, False]
    assert result.failure.tolist() == [False, True, False]
    assert result.timeout.tolist() == [False, False, True]
    assert result.reason.tolist() == [0, 3, 1]
    later = State(torch.zeros(3, dtype=torch.bool), torch.tensor([False, True, False]),
                  torch.tensor([True, False, True]), torch.tensor([1, 3, 1]))
    result, _ = player.continuation_state(later, reached)
    assert result.timeout.tolist() == [False, False, True]
    assert result.reason.tolist() == [0, 3, 1]


def test_temporary_ten_metre_landing_keeps_training_length():
    code = '''
import sys
sys.path.insert(0, 'scripts')
from a2_v07_bootstrap import activate
activate()
from play_a2_v09_s1 import configure_terrain
from src.tasks.velocity.v07 import spec, terrain
original = spec.terrain_spec(spec.tile_id_for('L40'))
played, width = configure_terrain({'geometry': {'tile_id': original.tile_id}},
                                 steps=20, width=2.0, length=10.0)
assert played.N == 20 and width == 2.0
assert played.platform_length == 10.0
assert played.valid_bounds_local[1] == 14.0
assert original.platform_length == 2.0 and original.N == 40
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_continuation_hooks_keep_s1_info_and_clear_latch_on_failure_reset():
    code = '''
import sys
sys.path.insert(0, 'scripts')
from a2_v07_bootstrap import activate
activate()
import torch
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch
from src.tasks.velocity.v07.env import V07Env
from play_a2_v09_s1 import enable_continue_after_success
@dataclass(frozen=True)
class State:
    success: object
    failure: object
    timeout: object
    reason: object
state = State(torch.tensor([True]), torch.tensor([False]),
              torch.tensor([False]), torch.tensor([2]))
resets = []
env = SimpleNamespace(v09_cases=[{'task':'stairs'}], num_envs=1, device='cpu',
                      _reset_ids=lambda ids, **kwargs: resets.append(ids.tolist()))
enable_continue_after_success(env)
with patch.object(V07Env, '_task_state', return_value=state):
    result = env._task_state()
assert not result.success.any() and not result.timeout.any()
assert env.playback_reached_top.tolist() == [True]
assert env._flat_outcome == {} and not env._physical_failure.any()
assert resets == []
env._reset_ids(torch.tensor([0]))
assert resets == [[0]] and not env.playback_reached_top.any()
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_spawn_seed_changes_pose_reproducibly_without_editing_case():
    from copy import deepcopy
    player = load_player()
    case = {'spawn': {'local_position_xy': [-1., 0.], 'local_yaw_radians': 0.,
                     'frame_origin_world': [0., 168., 0.],
                     'local_position': [-1., 0., .4], 'world_position': [-1., 168., .4]},
            'case_sha256': 'original', 'streams': {'spawn_x': 'frozen'}}
    original = deepcopy(case)
    a = player.resample_spawn(case, 43, x_bounds=(-1.1, -.9), y_bounds=(-.05, .05), yaw_bounds=(-.05, .05))
    b = player.resample_spawn(case, 43, x_bounds=(-1.1, -.9), y_bounds=(-.05, .05), yaw_bounds=(-.05, .05))
    c = player.resample_spawn(case, 44, x_bounds=(-1.1, -.9), y_bounds=(-.05, .05), yaw_bounds=(-.05, .05))
    assert a == b and a['spawn'] != c['spawn']
    assert case == original and a['case_sha256'] != 'original'
    assert a['playback_spawn_seed'] == 43
    assert -.05 <= a['spawn']['local_yaw_radians'] <= .05


def test_flat_walk_stop_walk_exact_control_boundaries():
    player = load_player()
    assert player.flat_sequence(399, 0.5, 8.0, 5.0) == ('WALK', 0.5)
    assert player.flat_sequence(400, 0.5, 8.0, 5.0) == ('STOP', 0.0)
    assert player.flat_sequence(649, 0.5, 8.0, 5.0) == ('STOP', 0.0)
    assert player.flat_sequence(650, 0.5, 8.0, 5.0) == ('WALK_AGAIN', 0.5)
    assert player.flat_sequence(2000, 0.5, 8.0, 5.0) == ('WALK_AGAIN', 0.5)


def test_flat_case_selection_does_not_pick_stop_or_stairs():
    player = load_player()
    cases = [dict(task='stairs', lesson='FLAT', speed=0.5, case_id=0),
             dict(task='flat_stop', lesson='FLAT', speed=0.5, case_id=0),
             dict(task='flat_walk', lesson='FLAT', speed=0.5, case_id=0)]
    assert player.select_case({'cases': cases}, 'FLAT', 0.5, 0, scene='flat') is cases[2]


def test_keyboard_walk_stop_resume_and_unrelated_keys():
    player = load_player()
    keys = player.KeyboardControl(0.5)
    assert keys.read() == ('STOP', 0.0)
    keys.on_key(ord('W'))
    assert keys.read() == ('WALK', 0.5)
    keys.on_key(ord('S'))
    assert keys.read() == ('STOP', 0.0)
    keys.on_key(ord('W'))
    assert keys.read() == ('WALK', 0.5)
    keys.on_key(ord('X'))
    assert keys.read() == ('WALK', 0.5)


def test_keyboard_drains_events_in_order():
    keys = load_player().KeyboardControl(0.5)
    for char in 'WsW':
        keys.on_key(ord(char))
    assert keys.read() == ('WALK', 0.5)
    keys.on_key(ord('s'))
    assert keys.read() == ('STOP', 0.0)
