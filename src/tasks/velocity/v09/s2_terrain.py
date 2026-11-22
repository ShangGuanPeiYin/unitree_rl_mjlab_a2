"""S2 real collision geometry and exact analytic surface queries (§4.1).

The geometry is a set of axis-aligned boxes identical to the collision shapes:

* approach/ground box: ``x in [-2, N*d + platform_length]``, ``|y| <= 2``, top 0;
* tread ``i = 1..N``: ``x in [(i-1)d, i d]``, ``|y| <= 0.80/2``, top ``i*h``;
* top platform: ``x in [N*d, N*d+20]``, ``|y| <= platform_width/2``, top ``N*h``.

The platform width is the real 0.8 m or 4 m row width; it is never clamped to
the stair width.  The last tread and the platform share the boundary ``x=N*d``
but do not overlap, so no extra side/support area is created.  The analytic
query returns the highest exact box top at ``(x, y)`` and a valid flag over the
frozen terrain bounds; outside the boxes the height is finite only inside those
bounds (otherwise ``nan``, the original convention), and no platform is
fabricated under a foot that is beside a narrow platform.

This module imports no simulator at import time.  :func:`compile_mujoco_model`
is the explicit CPU geometry-compilation step used by tests and, later, by the
physics env.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from . import s2_spec

FLAT_TILE_ID = 0
STAIR_GROUND_HALF_WIDTH = 2.0
GROUND_X_MIN = -2.0
STAIR_COUNTS = (40, 10, 20)
PLATFORM_WIDTHS = (0.8, 4.0)


@dataclass(frozen=True)
class S2Box:
    name: str
    step: int
    center: tuple[float, float, float]
    half: tuple[float, float, float]
    top: float

    @property
    def bounds(self):
        cx, cy, _ = self.center
        hx, hy, _ = self.half
        return (cx - hx, cx + hx, cy - hy, cy + hy)


@dataclass(frozen=True)
class S2TerrainSpec:
    terrain_key: str
    tile_id: int
    kind: str
    N: int
    h: float
    step_depth: float
    width: float
    tread_thickness: float
    platform_length: float
    platform_width: float
    frame_origin_world: tuple[float, float, float]
    valid_bounds_local: tuple[float, float, float, float]

    @property
    def platform_height(self):
        return self.N * self.h

    @property
    def platform_front_local_x(self):
        return self.N * self.step_depth


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a real number')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def terrain_specs(*, stair_counts=STAIR_COUNTS, platform_widths=PLATFORM_WIDTHS):
    """Stable S2 table; S2.0's 40-level rows come first so their origins match."""
    counts = tuple(int(v) for v in stair_counts)
    if any(v <= 0 for v in counts):
        raise ValueError('stair counts must be positive')
    widths = tuple(float(v) for v in platform_widths)
    if any(v not in PLATFORM_WIDTHS for v in widths):
        raise ValueError('platform widths must be 0.8 or 4.0')
    flat = S2TerrainSpec(
        terrain_key='s2_flat_v07_tile0', tile_id=FLAT_TILE_ID, kind='flat',
        N=0, h=0.0, step_depth=s2_spec.STAIR_DEPTH, width=8.0,
        tread_thickness=0.01, platform_length=0.0, platform_width=8.0,
        frame_origin_world=(0.0, 0.0, 0.0),
        valid_bounds_local=(-2.0, 22.0, -4.0, 4.0))
    rows = [flat]
    ordered = []
    for count in counts:
        for width in widths:
            ordered.append((count, width))
    # Put N=40 first to preserve the frozen S2.0 origins (12, 28).
    ordered.sort(key=lambda pair: (0 if pair[0] == 40 else 1, counts.index(pair[0]),
                                   widths.index(pair[1])))
    for index, (count, width) in enumerate(ordered):
        origin_y = 12.0 + 16.0 * index
        rows.append(S2TerrainSpec(
            terrain_key=f's2_stairs_h0.20_d0.20_w0.80_pw{width:g}',
            tile_id=1 + index, kind='s2_stairs', N=count,
            h=s2_spec.STAIR_HEIGHT, step_depth=s2_spec.STAIR_DEPTH,
            width=s2_spec.STAIR_WIDTH, tread_thickness=0.01,
            platform_length=s2_spec.PLATFORM_LENGTH, platform_width=width,
            frame_origin_world=(0.0, origin_y, 0.0),
            valid_bounds_local=(GROUND_X_MIN, count * s2_spec.STAIR_DEPTH
                                + s2_spec.PLATFORM_LENGTH, -2.0, 2.0)))
    return tuple(rows)


def _match_table_spec(geometry):
    for spec in terrain_specs():
        if spec.kind != ('flat' if geometry.get('kind') == 'flat' else 's2_stairs'):
            continue
        if geometry.get('kind') == 'flat':
            return spec
        if (int(geometry['N']) == spec.N
                and abs(float(geometry['platform_width']) - spec.platform_width) < 1e-12
                and abs(float(geometry['h']) - spec.h) < 1e-12
                and abs(float(geometry['step_depth']) - spec.step_depth) < 1e-12
                and abs(float(geometry['platform_length']) - spec.platform_length) < 1e-12):
            return spec
    return None


def schedule_tile_id(geometry):
    """Explicit S2 terrain_key -> legal schedule tile id (0..len(table)-1)."""
    spec = _match_table_spec(geometry)
    if spec is None:
        raise ValueError(f'no S2 schedule row for geometry {geometry.get("terrain_key")}')
    return int(spec.tile_id)


def schedule_tile_table():
    table = {(spec.terrain_key, spec.N, spec.platform_width): int(spec.tile_id)
             for spec in terrain_specs()}
    ids = list(table.values())
    if len(ids) != len(set(ids)):
        raise ValueError('S2 schedule tile ids are not unique')
    return table


def spec_from_geometry(geometry):
    """Resolve a frozen case geometry dict to the exact S2 collision row."""
    if not isinstance(geometry, dict):
        raise ValueError('geometry must be a mapping')
    kind = geometry.get('kind')
    if kind == 'flat':
        return terrain_specs()[0]
    if kind != 's2_stairs':
        raise ValueError(f'unknown S2 geometry kind: {kind}')
    N = int(geometry['N'])
    width = float(geometry['platform_width'])
    if width not in PLATFORM_WIDTHS:
        raise ValueError('platform width must be 0.8 or 4.0')
    if abs(float(geometry['h']) - s2_spec.STAIR_HEIGHT) > 1e-12:
        raise ValueError('S2 stair height must be 0.20 m')
    if abs(float(geometry['step_depth']) - s2_spec.STAIR_DEPTH) > 1e-12:
        raise ValueError('S2 stair depth must be 0.20 m')
    if abs(float(geometry['platform_length']) - s2_spec.PLATFORM_LENGTH) > 1e-12:
        raise ValueError('S2 platform length must be 20 m')
    origin = tuple(float(v) for v in geometry['frame_origin_world'])
    bounds = tuple(float(v) for v in geometry['valid_bounds_local'])
    matched = _match_table_spec(geometry)
    tile_id = int(matched.tile_id) if matched is not None else schedule_tile_id(geometry)
    return S2TerrainSpec(
        terrain_key=geometry.get('terrain_key', f's2_stairs_N{N}_pw{width:g}'),
        tile_id=tile_id, kind='s2_stairs', N=N,
        h=float(geometry['h']), step_depth=float(geometry['step_depth']),
        width=float(geometry['width']), tread_thickness=float(geometry['tread_thickness']),
        platform_length=float(geometry['platform_length']),
        platform_width=width, frame_origin_world=origin,
        valid_bounds_local=bounds)


def legacy_spec_from_geometry(geometry):
    """Original v07 H/L geometry (2 m platform, non-0.20 step height)."""
    N = int(geometry['N'])
    h = float(geometry['h'])
    d = float(geometry['step_depth'])
    width = float(geometry['width'])
    platform_length = float(geometry['platform_length'])
    origin = tuple(float(v) for v in geometry['frame_origin_world'])
    bounds = tuple(float(v) for v in geometry['valid_bounds_local'])
    if N <= 0 or h <= 0 or d <= 0 or width <= 0 or platform_length <= 0:
        raise ValueError('invalid legacy stair geometry')
    return S2TerrainSpec(
        terrain_key=f"legacy_{geometry.get('lesson_id', 'unknown')}",
        tile_id=int(geometry.get('tile_id') or -1), kind='s2_stairs', N=N, h=h,
        step_depth=d, width=width, tread_thickness=float(geometry['tread_thickness']),
        platform_length=platform_length, platform_width=width,
        frame_origin_world=origin, valid_bounds_local=bounds)


def boxes(spec):
    """Exact world-axis-aligned collision boxes (ground first, then treads, platform)."""
    ox, oy, oz = (float(v) for v in spec.frame_origin_world)
    if spec.kind == 'flat':
        x0, x1, y0, y1 = spec.valid_bounds_local
        return (S2Box('flat', 0, (ox + (x0 + x1) / 2.0, oy + (y0 + y1) / 2.0, oz - 0.05),
                      ((x1 - x0) / 2.0, (y1 - y0) / 2.0, 0.05), oz),)
    n, h, d = int(spec.N), float(spec.h), float(spec.step_depth)
    half_t = float(spec.tread_thickness) / 2.0
    length = n * d + float(spec.platform_length) + 2.0
    result = [S2Box('ground', 0,
                    (ox - 2.0 + length / 2.0, oy, oz - 0.05),
                    (length / 2.0, STAIR_GROUND_HALF_WIDTH, 0.05), oz)]
    for i in range(1, n + 1):
        result.append(S2Box(
            f'tread_{i}', i, (ox + (i - 0.5) * d, oy, oz + i * h - half_t),
            (d / 2.0, float(spec.width) / 2.0, half_t), oz + i * h))
    result.append(S2Box(
        'platform', n, (ox + n * d + float(spec.platform_length) / 2.0, oy,
                        oz + n * h - half_t),
        (float(spec.platform_length) / 2.0, float(spec.platform_width) / 2.0, half_t),
        oz + n * h))
    return tuple(result)


def query_surface(points_world, spec):
    """Highest exact surface at ``(x,y)``; ``nan`` outside the frozen bounds."""
    points = np.asarray(points_world, dtype=np.float64)
    if points.shape[-1] not in (2, 3):
        raise ValueError('points must have last axis 2 or 3')
    x0, x1, y0, y1 = (float(v) for v in spec.valid_bounds_local)
    ox, oy, oz = (float(v) for v in spec.frame_origin_world)
    x = points[..., 0]
    y = points[..., 1]
    valid = ((x >= x0 + ox) & (x <= x1 + ox)
             & (y >= y0 + oy) & (y <= y1 + oy))
    height = np.full(x.shape, -np.inf, dtype=np.float64)
    for box in boxes(spec):
        bx0, bx1, by0, by1 = box.bounds
        inside = (x >= bx0) & (x <= bx1) & (y >= by0) & (y <= by1)
        height = np.where(inside & (box.top > height), box.top, height)
    height = np.where(valid, height, np.nan)
    return height, valid


def query_surface_height(points_world, spec):
    return query_surface(points_world, spec)[0]


def surface_class(points_world, *, spec):
    """Classify points as ground/tread/platform/outside for support queries."""
    points = np.asarray(points_world, dtype=np.float64)
    ox, oy, _ = (float(v) for v in spec.frame_origin_world)
    x = points[..., 0] - ox
    y = points[..., 1] - oy
    valid = ((x >= spec.valid_bounds_local[0]) & (x <= spec.valid_bounds_local[1])
             & (y >= spec.valid_bounds_local[2]) & (y <= spec.valid_bounds_local[3]))
    result = np.full(x.shape, 'outside', dtype=object)
    if spec.kind == 'flat':
        result[valid] = 'ground'
        return result
    n, d = spec.N, spec.step_depth
    stair_end = n * d
    result[valid] = 'ground'
    tread = (valid & (x >= 0.0) & (x < stair_end)
             & (np.abs(y) <= float(spec.width) / 2.0))
    result[tread] = 'tread'
    platform = (valid & (x >= stair_end)
                & (x <= stair_end + float(spec.platform_length))
                & (np.abs(y) <= float(spec.platform_width) / 2.0))
    result[platform] = 'platform'
    return result


def platform_bounds(spec):
    x0 = float(spec.frame_origin_world[0]) + float(spec.N) * float(spec.step_depth)
    x1 = x0 + float(spec.platform_length)
    y = float(spec.frame_origin_world[1])
    half = float(spec.platform_width) / 2.0
    return (x0, x1, y - half, y + half)


def build_mjcf(specs):
    """MJCF with one box geom per exact collision box (no physics run)."""
    geoms = []
    for spec in specs:
        for box in boxes(spec):
            cx, cy, cz = box.center
            hx, hy, hz = box.half
            geoms.append(
                f'<geom name="s2_{spec.tile_id:02d}_{box.name}" type="box" '
                f'pos="{cx:.12g} {cy:.12g} {cz:.12g}" '
                f'size="{hx:.12g} {hy:.12g} {hz:.12g}" '
                f'contype="1" conaffinity="1" condim="3" '
                f'friction="0.6 0.005 0.0001"/>')
    return ('<mujoco model="s2_terrain"><worldbody><body name="terrain">'
            + ''.join(geoms) + '</body></worldbody></mujoco>')


def compile_mujoco_model(specs):
    """CPU-only compile of the exact collision boxes (no `mj_step`)."""
    import mujoco
    return mujoco.MjModel.from_xml_string(build_mjcf(specs))


def validate_compiled_model(model, specs, *, atol=1e-9):
    expected = [(spec, box) for spec in specs for box in boxes(spec)]
    if int(model.ngeom) != len(expected):
        raise ValueError(f'compiled geom count {model.ngeom} != expected {len(expected)}')
    for geom_id, (spec, box) in enumerate(expected):
        pos = np.asarray(model.geom_pos[geom_id], dtype=np.float64)
        size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
        if not (np.allclose(pos, box.center, rtol=0, atol=atol)
                and np.allclose(size, box.half, rtol=0, atol=atol)):
            raise ValueError(f'compiled geometry differs at geom {geom_id} ({spec.terrain_key}/{box.name})')
    return True


def metadata(specs):
    return dict(schema_version=s2_spec.CHECKPOINT_SCHEMA, stage=s2_spec.STAGE,
                sds_sha256=s2_spec.SDS_SHA256,
                specs=[asdict(spec) for spec in specs],
                boxes=[dict(name=box.name, step=box.step, center=list(box.center),
                            half=list(box.half), top=box.top)
                       for spec in specs for box in boxes(spec)])
