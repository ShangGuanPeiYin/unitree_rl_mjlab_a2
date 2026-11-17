"""v07 native recipe, appended physical D22 and unchanged C1 training settings."""
from dataclasses import asdict
from pathlib import Path
import os
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v07_bootstrap

a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v07 import spec


def test_recipe_consumers_use_native_v07_identity_and_budget():
    recipe = spec.recipe_config(max_updates=48000,
        phase_budgets='F0=1000,H=12000,D=21000,L=6000,V=4000,R=4000')
    assert recipe['recipe'] == '0.7-d22-eval-regression-r1'
    assert spec.identity_block()['code_version'] == '0.7.0'
    assert spec.identity_block()['schema_version'] == 9
    assert spec.ARCHITECTURE == 'v06-route-state-v1'
    assert spec.PHASE_BY_ID['D'].substages == ('D25', 'D22', 'D20')
    assert spec.PHASE_BY_ID['D'].substage_budgets == (6000, 6000, 9000)
    assert spec.total_budget() == 48000
    assert spec.TOTAL_CONTROL_SAMPLES == 1_179_648_000
    assert spec.TOTAL_PHYSICS_SUBSTEPS == 4_718_592_000
    assert spec.TOTAL_ADAM_UPDATES == 960_000
    with pytest.raises(ValueError):
        spec.recipe_config(max_updates=36000)


def test_original_physical_tiles_do_not_move_and_d22_is_appended():
    from src.tasks.velocity.v06 import spec as original
    assert [asdict(tile) for tile in spec.TERRAIN_TABLE[:22]] == [
        asdict(tile) for tile in original.TERRAIN_TABLE]
    tile = spec.terrain_spec(22)
    assert asdict(tile) == dict(tile_id=22, lesson_id='D22', N=6, h=.2,
        step_depth=.22, width=.8, tread_thickness=.01, platform_length=2.,
        frame_origin_world=(0., 176., 0.), valid_bounds_local=(-2., 3.3200000000000003, -2., 2.))
    assert spec.tile_id_for('D22') == 22
    assert spec.terrain_spec(spec.FINAL_TILE_ID).lesson_id == 'L40'
    assert tuple(spec.CURRICULUM_TILE_ORDER) == tuple(range(1, 18)) + (22, 18, 19, 20, 21)
    assert len(spec.TERRAIN_TABLE) == spec.PHYSICAL_TILE_COUNT == 23
    with pytest.raises(ValueError):
        spec.terrain_spec(23)


def test_c1_ppo_rewards_spawn_and_command_contract_are_preserved():
    from src.tasks.velocity.v06 import spec as original
    assert spec.PPO_CONFIG == original.PPO_CONFIG
    before, after = original.reward_config(), spec.reward_config()
    before.pop('version')
    after.pop('version')
    assert before == after
    assert (spec.COMMAND_INITIAL_VX, spec.COMMAND_FINAL_VX) == ((.3,.6),(.3,.6))
    assert (spec.COMMAND_VY, spec.COMMAND_WZ) == (0.,0.)
    assert (spec.SPAWN_X,spec.SPAWN_Y,spec.SPAWN_YAW_DEGREES) == ((-1.1,-.9),(-.05,.05),(-5.,5.))
    assert all(p.gate_speeds == (.3,.5,.6) and p.reward_speeds == (.3,.6)
               for p in spec.PHASES)
    assert spec.RESOURCE_LIMITS == original.RESOURCE_LIMITS


def test_v07_import_ignores_legacy_c1_selection_and_does_not_import_v06_spec():
    code = '''
import sys
sys.path.insert(0, 'scripts')
import a2_v07_bootstrap
a2_v07_bootstrap.activate()
from src.tasks.velocity.v07 import spec
assert spec.recipe_config()['max_updates'] == 48000
assert spec.SCHEMA_VERSION == 9
assert 'src.tasks.velocity.v06.spec' not in sys.modules
'''
    env = dict(os.environ, A2_V06_CONTINUATION='unknown-hostile-legacy-contract')
    result = subprocess.run([sys.executable,'-c',code], cwd=ROOT, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_frozen_document_contract_points_to_real_unchanged_sources():
    for _,relative,digest in spec.FROZEN_DOCUMENTS + spec.SUPPLEMENTARY_DOCUMENTS:
        assert (ROOT/relative).is_file(), relative
        assert spec.sha256(ROOT/relative) == digest
