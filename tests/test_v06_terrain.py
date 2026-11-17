"""SDS 9.1 items 11/12: real v0.6 terrain geometry, metadata, map and bounds.

Every assertion below is made against the *compiled MuJoCo model* (box world
coordinates and ``mj_ray``) and against the exact surface query, never against a
configuration string alone.  All 22 tiles are covered.
"""
import math

import mujoco
import numpy as np
import pytest
import torch

from src.tasks.velocity.v06 import spec, terrain


@pytest.fixture(scope='module')
def build():
    return terrain.build_terrain(seed=42)


def _ray(model, data, x, y, z0=12.0):
    gid = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(model, data, np.array([x, y, z0], dtype=np.float64),
                             np.array([0.0, 0.0, -1.0]), None, True, -1, gid)
    return distance, int(gid[0])


def test_table_is_22_tiles_and_matches_sds_heights_and_depths():
    table = terrain.build_terrain_specs()
    assert table == tuple(spec.TERRAIN_TABLE)
    assert len(table) == 22
    assert [s.lesson_id for s in table] == ['FLAT'] + [row[0] for row in spec.TERRAIN_ROWS]
    for index, row in enumerate(spec.TERRAIN_ROWS):
        tile = table[index + 1]
        assert int(tile.N) == row[1]
        assert math.isclose(float(tile.h), row[2], abs_tol=1e-12)
        assert math.isclose(float(tile.step_depth), row[3], abs_tol=1e-12)
        assert tile.width == 0.80 and tile.tread_thickness == 0.01
        assert tile.platform_length == 2.0
    # H00..H12 heights and the fixed H13/H14/H15 + D/L depths.
    assert [round(float(t.h), 12) for t in table[1:14]] == [
        round(0.02 + 0.13 * j / 12.0, 12) for j in range(13)]
    assert [float(t.h) for t in table[14:17]] == [0.16, 0.18, 0.20]
    assert [float(t.step_depth) for t in table[18:22]] == [0.20] * 4
    assert [int(t.N) for t in table[18:22]] == [6, 12, 20, 40]
    for tile in table[1:]:
        bounds = tuple(float(v) for v in tile.valid_bounds_local)
        assert bounds == (-2.0, int(tile.N) * float(tile.step_depth) + 2.0, -2.0, 2.0)
    assert tuple(table[0].valid_bounds_local) == (-2.0, 22.0, -4.0, 4.0)


def test_compiled_boxes_are_the_exact_spec_geometry(build):
    index = build.index
    model = build.model
    specs = build.specs
    for tile in specs:
        tid = int(tile.tile_id)
        expected = terrain.tile_boxes(tile)
        geoms = build.tile_geoms(tid)
        assert len(geoms) == len(expected) == (1 if int(tile.N) == 0 else int(tile.N) + 2)
        got = sorted((tuple(np.round(model.geom_pos[g], 9)),
                      tuple(np.round(model.geom_size[g], 9))) for g in geoms)
        wanted = sorted((tuple(np.round(b.center, 9)), tuple(np.round(b.half, 9)))
                        for b in expected)
        assert got == wanted
        for g in geoms:
            assert int(index.tile[g]) == tid
            assert int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_BOX)
            assert math.isclose(float(model.geom_pos[g][2] + model.geom_size[g][2]),
                                float(index.top[g]), abs_tol=1e-9)
        # The 0.80 m width is the real footprint of treads and platform.
        for g in geoms:
            if int(index.step[g]) > 0:
                assert math.isclose(float(model.geom_size[g][1]), 0.4, abs_tol=1e-12)
                assert math.isclose(float(model.geom_size[g][2]), 0.005, abs_tol=1e-12)


def test_query_matches_mj_ray_on_all_22_tiles(build):
    model, data = build.model, mujoco.MjData(build.model)
    mujoco.mj_forward(model, data)
    for tile in build.specs:
        tid = int(tile.tile_id)
        ox, oy, _ = (float(v) for v in tile.frame_origin_world)
        n, h, d = int(tile.N), float(tile.h), float(tile.step_depth)
        xs = [-1.5, -0.5, -0.05]
        if n == 0:
            xs += [0.5, 10.0, 21.5]
        else:
            xs += [(i - 0.5) * d for i in range(1, n + 1)]
            xs += [n * d + 0.5, n * d + 1.99]
        for x in xs:
            for y in (0.0, 0.3, -0.3):
                distance, gid = _ray(model, data, ox + x, oy + y)
                assert distance >= 0.0
                ray_z = 12.0 - distance
                query_z = float(terrain.query_surface_height(
                    torch.tensor([[ox + x, oy + y, 0.0]]), torch.tensor([tid]))[0])
                assert math.isclose(query_z, ray_z, abs_tol=2e-6), (tid, x, y)
                assert int(build.index.tile[gid]) == tid
        # The flat/approach ground outside the stair width is still the exact
        # approach box (no side walls): y beyond 0.4 on stairs lands on ground.
        if n:
            query_z = float(terrain.query_surface_height(
                torch.tensor([[ox + 0.5, oy + 1.5, 0.0]]), torch.tensor([tid]))[0])
            assert query_z == 0.0


def test_open_risers_and_thin_treads_have_no_solid_riser(build):
    model, data = build.model, mujoco.MjData(build.model)
    mujoco.mj_forward(model, data)
    for tile in build.specs[1:]:
        ox, oy, _ = (float(v) for v in tile.frame_origin_world)
        h = float(tile.h)
        # Half a tread height above the first tread front edge: no riser/sidewall.
        gid = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(
            model, data, np.array([ox, oy, h / 2.0 - 0.002]),
            np.array([1.0, 0.0, 0.0]), None, True, -1, gid)
        assert distance == -1, tile.lesson_id


def test_step_depth_is_never_the_old_020_constant(build):
    index = build.index
    model = build.model
    h00 = spec.terrain_spec(spec.tile_id_for('H00'))
    assert float(h00.step_depth) == 0.30
    # The first tread of H00 is 0.30 m deep, so its front edge sits at x=0 and
    # the platform edge is N*0.30 = 1.80 m (never N*0.20 = 1.20 m).
    boxes = {box.name: box for box in terrain.tile_boxes(h00)}
    tread1 = boxes['tread_1']
    assert math.isclose(tread1.center[0] - h00.frame_origin_world[0], 0.15, abs_tol=1e-12)
    assert math.isclose(tread1.bounds[1] - tread1.bounds[0], 0.30, abs_tol=1e-12)
    assert math.isclose(terrain.platform_start(int(h00.tile_id)), 1.80, abs_tol=1e-12)
    x0, x1, y0, y1 = terrain.platform_bounds(int(h00.tile_id))
    assert math.isclose(x0, 1.80 + spec.SAFE_INSET, abs_tol=1e-12)
    assert math.isclose(x1, 1.80 + 2.0 - spec.SAFE_INSET, abs_tol=1e-12)
    assert math.isclose(y0, -0.40 + spec.SAFE_INSET, abs_tol=1e-12)
    # D25 / D20 / L40 really use their own depth in geometry and bounds.
    for lesson, depth in (('D25', 0.25), ('D20', 0.20), ('L40', 0.20)):
        tile = spec.terrain_spec(spec.tile_id_for(lesson))
        assert float(tile.step_depth) == depth
        assert math.isclose(tuple(tile.valid_bounds_local)[1],
                            int(tile.N) * depth + 2.0, abs_tol=1e-12)
        tid = int(tile.tile_id)
        step1 = [g for g in build.tile_geoms(tid)
                 if int(index.step[g]) == 1][0]
        assert math.isclose(float(model.geom_size[step1][0]), depth / 2.0, abs_tol=1e-12)
        assert math.isclose(float(terrain.platform_bounds(tid)[0]),
                            int(tile.N) * depth + spec.SAFE_INSET, abs_tol=1e-12)
    # A 0.20-based height query would return the wrong top for H00 tread 1.
    ox, oy, _ = (float(v) for v in h00.frame_origin_world)
    for x in (0.05, 0.25):
        z = float(terrain.query_surface_height(
            torch.tensor([[ox + x, oy, 0.0]]), torch.tensor([int(h00.tile_id)]))[0])
        assert z == pytest.approx(float(h00.h)), x
    # x = 0.25 lies in the old 0.20 grid's second tread but still in tread 1 here.
    assert math.isclose(float(h00.h), 0.02, abs_tol=1e-12)


def test_valid_bounds_and_platform_safe_area(build):
    for tile in build.specs:
        tid = int(tile.tile_id)
        ox, oy, _ = (float(v) for v in tile.frame_origin_world)
        x0, x1, y0, y1 = (float(v) for v in tile.valid_bounds_local)
        inside = torch.tensor([[ox + x0 + 0.01, oy + y0 + 0.01, 0.0],
                               [ox + x1 - 0.01, oy + y1 - 0.01, 0.0]])
        height, valid = terrain.query_surface(inside, torch.tensor([tid, tid]))
        assert bool(valid.all()) and torch.isfinite(height).all()
        outside = torch.tensor([[ox + x0 - 0.01, oy, 0.0],
                                [ox, oy + y1 + 0.01, 0.0]])
        height, valid = terrain.query_surface(outside, torch.tensor([tid, tid]))
        assert not bool(valid.any())
        assert torch.isnan(height).all()
        if int(tile.N):
            px0, px1, py0, py1 = terrain.platform_bounds(tid)
            assert px0 >= int(tile.N) * float(tile.step_depth)
            assert px1 <= int(tile.N) * float(tile.step_depth) + 2.0
            assert px1 > px0 and py1 > py0
        else:
            assert all(math.isnan(v) for v in terrain.platform_bounds(tid))


def test_timeout_uses_current_depth_and_sds_formula():
    for tile in spec.TERRAIN_TABLE[1:]:
        n, h, d = int(tile.N), float(tile.h), float(tile.step_depth)
        command = torch.tensor([0.5])
        got = int(terrain.timeout_steps(int(tile.tile_id), command)[0])
        expected = math.ceil((2.0 * (2.2 + n * math.sqrt(h * h + d * d)) / 0.5 + 10.0)
                             / float(spec.CONTROL_DT))
        assert got == expected
    assert int(terrain.timeout_steps(0, torch.tensor([0.5]))[0]) == spec.F0_EPISODE_STEPS
    # H00 (d=0.30) and D20 (d=0.20) differ: the old 0.20 constant is not used.
    h00 = int(terrain.timeout_steps(spec.tile_id_for('H00'), torch.tensor([0.5]))[0])
    h00_old = math.ceil((2.0 * (2.2 + 6 * math.sqrt(0.02 ** 2 + 0.2 ** 2)) / 0.5 + 10.0)
                        / 0.02)
    assert h00 != h00_old


def test_geometry_index_has_exact_landing_support_rectangles(build):
    index = build.index
    for tile in build.specs:
        tid = int(tile.tile_id)
        n = int(tile.N)
        if not n:
            continue
        last = np.flatnonzero((index.tile == tid) & (index.step == n))
        assert len(last) == 2
        x0 = tile.frame_origin_world[0] + (n - 1) * float(tile.step_depth)
        x1 = (tile.frame_origin_world[0] + n * float(tile.step_depth)
              + float(tile.platform_length))
        for g in last:
            np.testing.assert_allclose(
                index.support_bounds[g],
                [x0, x1, tile.frame_origin_world[1] - 0.4,
                 tile.frame_origin_world[1] + 0.4], atol=1e-9)


def test_map_samples_only_v06_terrain():
    from src.tasks.velocity.v06.map import HeightMap, MAP_REFRESH_STEPS

    cache = HeightMap(2, 'cpu')
    tile = spec.terrain_spec(spec.tile_id_for('L20'))
    base = torch.tensor([[tile.frame_origin_world[0] - 0.9,
                          tile.frame_origin_world[1], 0.4],
                         [0.0, 0.0, 0.4]], dtype=torch.float32)
    tiles = torch.tensor([int(tile.tile_id), 0])
    out = cache.reset(base, torch.zeros(2), tiles)
    assert tuple(out.shape) == (2, 3, 48, 32)
    # The map is the exact query relative to the base, never an atlas lookup.
    points = cache.sample_points_world[0]
    height, valid = terrain.query_surface(points, torch.tensor([int(tile.tile_id)]))
    expected = torch.where(valid, ((height - base[0, 2]) / 1.5).clamp(-1, 1),
                           torch.zeros_like(height))
    torch.testing.assert_close(cache.map[0, 0], expected)
    assert int(cache.sample_tile[0]) == int(tile.tile_id)
    # Flat tile has exactly zero height everywhere inside its bounds.
    assert torch.all(cache.map[1, 0] <= 0.0)
    # 10 Hz: the height channel is held for four further control steps.
    held = cache.map[0, 0].clone()
    cache.update(base + torch.tensor([[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]]),
                 torch.zeros(2), tiles)
    torch.testing.assert_close(cache.map[0, 0], held)
    for _ in range(MAP_REFRESH_STEPS - 1):
        cache.update(base, torch.zeros(2), tiles)
    assert int(cache.steps[0]) == 0
