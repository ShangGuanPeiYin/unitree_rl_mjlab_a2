"""Current case tables route D22, final L40 and retention independently."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import a2_v07_bootstrap

a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v07 import evaluation, spec


@pytest.mark.parametrize('lesson,tile', [('D25',17),('D22',22),('D20',18)])
def test_current_difficulty_case_table_uses_physical_tile_not_sorted_position(lesson,tile):
    table = evaluation.make_cases('D',lesson,'validation',42)
    assert table.target_tile_id == tile
    assert table.target_lesson == lesson
    assert table.substage == lesson
    assert len(table.cases) == 60
    assert tuple(table.speeds) == (.3,.5,.6)
    assert all(case['tile_id'] == tile for case in table.cases)
    reloaded = evaluation.table_from_json(table.to_json(),expected_sha=table.sha256)
    assert reloaded.target_tile_id == tile
    assert reloaded.sha256 == table.sha256


@pytest.mark.parametrize('kind,tile,measurement', [('flat_retention',0,'flat'),('stair_retention',1,'stair')])
def test_d22_retention_routes_remain_pinned_to_original_terrains(kind,tile,measurement):
    table = evaluation.make_cases('D','D22',kind,42)
    assert table.target_tile_id == tile
    assert table.measurement == measurement
    assert len(table.cases) == 15
    assert tuple(table.speeds) == (.3,.5,.6)
    assert all(case['tile_id'] == tile for case in table.cases)


def test_final_route_stays_l40_tile21_despite_appended_maximum_tile22():
    table = evaluation.make_cases('R',None,'test',42)
    assert spec.FINAL_TILE_ID == table.target_tile_id == 21
    assert table.target_lesson == 'L40'
    assert len(table.cases) == 300
    assert all(case['tile_id'] == 21 for case in table.cases)
    with pytest.raises(evaluation.EvaluationError):
        evaluation.make_cases('R',None,'test',42,tile_id=22)
