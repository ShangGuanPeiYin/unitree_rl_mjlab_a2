"""CPU-only S2 terrain geometry and analytic query tests."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v07_bootstrap

a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v09 import s2_cases, s2_terrain as terrain


def spec(N=40, width=0.8):
    return next(row for row in terrain.terrain_specs()
                if row.N == N and row.platform_width == width
                and row.kind == 's2_stairs')


def test_table_matches_frozen_s2_0_origins_and_heights():
    rows = terrain.terrain_specs()
    assert len(rows) == 7
    flat = rows[0]
    assert flat.kind == 'flat' and flat.N == 0
    assert flat.frame_origin_world == (0.0, 0.0, 0.0)
    for width, origin_y in ((0.8, 12.0), (4.0, 28.0)):
        row = next(item for item in rows
                   if item.N == 40 and item.platform_width == width)
        assert row.frame_origin_world == (0.0, origin_y, 0.0)
        assert row.platform_height == pytest.approx(8.0)
        assert row.platform_front_local_x == pytest.approx(8.0)
        assert row.valid_bounds_local == (-2.0, 28.0, -2.0, 2.0)
    assert spec(10, 0.8).platform_height == pytest.approx(2.0)
    assert spec(20, 4.0).platform_height == pytest.approx(4.0)


def test_boxes_are_exact_and_seam_does_not_overlap():
    row = spec(40, 0.8)
    boxes = terrain.boxes(row)
    assert len(boxes) == 42
    assert boxes[0].name == 'ground'
    assert boxes[0].bounds == (-2.0, 28.0, 10.0, 14.0)  # origin y=12, ground half 2
    last_tread = boxes[40]
    platform = boxes[41]
    assert last_tread.bounds == pytest.approx((7.8, 8.0, 11.6, 12.4))
    assert platform.bounds == pytest.approx((8.0, 28.0, 11.6, 12.4))
    assert last_tread.bounds[1] == platform.bounds[0]
    # No box claims area outside the platform width or the ground width.
    assert platform.bounds[2] == pytest.approx(12.0 - 0.4)
    assert platform.bounds[3] == pytest.approx(12.0 + 0.4)


def test_query_ground_tread_platform_and_outside():
    row = spec(40, 0.8)
    points = np.array([
        [-1.0, 12.0], [0.1, 12.0], [7.9, 12.0], [8.0, 12.0], [27.9, 12.0],
        [28.1, 12.0], [8.0, 13.0], [0.1, 13.0]])
    height, valid = terrain.query_surface(points, row)
    assert height[0] == pytest.approx(0.0)
    assert height[1] == pytest.approx(0.2)
    assert height[2] == pytest.approx(8.0)
    assert height[3] == pytest.approx(8.0)
    assert height[4] == pytest.approx(8.0)
    assert np.isnan(height[5]) and valid[5] is np.False_
    # Outside the 0.8 m stair/platform width only the ground exists.
    assert height[6] == pytest.approx(0.0)
    assert height[7] == pytest.approx(0.0)
    assert valid.all() or not valid[5]


def test_wide_platform_is_not_clamped_to_narrow_width():
    narrow = spec(40, 0.8)
    wide = spec(40, 4.0)
    local = np.array([[8.0, 0.0], [8.0, 1.0], [8.0, 1.9], [8.0, 2.1]])
    narrow_points = local + np.array([0.0, narrow.frame_origin_world[1]])
    wide_points = local + np.array([0.0, wide.frame_origin_world[1]])
    narrow_h, narrow_valid = terrain.query_surface(narrow_points, narrow)
    wide_h, wide_valid = terrain.query_surface(wide_points, wide)
    assert narrow_h[0] == pytest.approx(8.0)
    assert narrow_h[1] == pytest.approx(0.0)
    assert wide_h[1] == pytest.approx(8.0)
    assert wide_h[2] == pytest.approx(8.0)
    assert np.isnan(wide_h[3]) and wide_valid[3] is np.False_
    assert terrain.platform_bounds(wide)[1] - terrain.platform_bounds(wide)[0] == pytest.approx(20.0)
    assert terrain.platform_bounds(wide)[3] - terrain.platform_bounds(wide)[2] == pytest.approx(4.0)


def test_surface_class_separates_ground_tread_and_platform():
    row = spec(40, 4.0)
    points = np.array([[-1.0, 0.2], [0.1, 0.2], [0.1, 1.0],
                       [7.9, 0.2], [8.0, 1.0], [28.1, 0.2]])
    points[:, 1] += row.frame_origin_world[1]
    classes = terrain.surface_class(points, spec=row)
    assert classes[0] == 'ground'
    assert classes[1] == 'tread'
    assert classes[2] == 'ground'       # outside the 0.8 m stair width
    assert classes[3] == 'tread'
    assert classes[4] == 'platform'
    assert classes[5] == 'outside'


def test_spec_from_frozen_case_geometry_matches_table():
    case = s2_cases.build_s2_0_manifest()['cases'][0]
    row = terrain.spec_from_geometry(case['geometry'])
    assert row.kind == 'flat'
    stair_case = next(item for item in s2_cases.build_s2_0_manifest()['cases']
                      if item['task'] == 'stairs_walk')
    stair = terrain.spec_from_geometry(stair_case['geometry'])
    assert stair.N == 40 and stair.platform_width in (0.8, 4.0)
    assert stair.frame_origin_world == tuple(stair_case['geometry']['frame_origin_world'])


def test_legacy_h_geometry_uses_original_platform_length():
    record = s2_cases.build_h_series_case0_manifest(ROOT)['records'][2]
    row = terrain.legacy_spec_from_geometry(record['case']['geometry'])
    assert row.N == 6
    assert row.platform_length == pytest.approx(2.0)
    assert row.h == pytest.approx(record['case']['geometry']['h'])
    origin_y = row.frame_origin_world[1]
    h, valid = terrain.query_surface(
        np.array([[1.7, origin_y], [1.9, origin_y], [3.7, origin_y]]), row)
    assert h[0] == pytest.approx(6 * row.h)   # platform begins at 6*0.30=1.8
    assert h[1] == pytest.approx(6 * row.h)
    assert h[2] == pytest.approx(6 * row.h)   # platform spans to 1.8+2.0


def test_mujoco_compile_matches_exact_boxes():
    pytest.importorskip('mujoco')
    rows = (spec(40, 0.8), spec(40, 4.0))
    model = terrain.compile_mujoco_model(rows)
    assert terrain.validate_compiled_model(model, rows) is True
    assert int(model.ngeom) == sum(len(terrain.boxes(row)) for row in rows)
