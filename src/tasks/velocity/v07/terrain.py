"""v0.7 real open-stair terrain: 23 tiles (flat + ``T0..T21``) as exact boxes.

SDS v0.7 sections 5.2 and 7.1.  This module is the *single* v0.7 geometry
source.  It does not import, reuse or extend the v0.2 atlas / ``v02.spec.STAGES``
path in any way: every tile is built from :data:`spec.TERRAIN_TABLE` (``TerrainSpec``)
and every other v0.7 module (map, contacts, events, rewards height lookup,
success criteria, timeout) consumes the same table through this module.

Geometry (per tile, tile-local coordinates with ``x = 0`` at the front edge of
the first tread, i.e. ``frame_origin_world``):

* approach ground box: ``x in [-2, N*d + platform_length]``, ``|y| <= 2``, top ``z = 0``
  (flat tiles use the full 24 x 8 tile, ``x in [-2, 22]``, ``|y| <= 4``);
* tread ``i = 1..N``: half size ``(d/2, width/2, tread_thickness/2)``, centre
  ``x = (i - 1/2) d``, top ``z = i*h`` -- open riser, no stringer, no rail;
* top platform: half size ``(platform_length/2, width/2, tread_thickness/2)``,
  centre ``x = N*d + platform_length/2``, top ``z = N*h``.

All boxes are axis aligned and *exactly* the collision geometry.  The surface
query is analytic over these same boxes (highest top face at the queried
``(x, y)``), so it can never disagree with the collision geometry and it is
never quantised to a height-field resolution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import mujoco
import numpy as np
import torch
from mjlab.terrains.terrain_generator import (
    SubTerrainCfg,
    TerrainGenerator,
    TerrainGeneratorCfg,
    TerrainGeometry,
    TerrainOutput,
)

from . import spec as _spec

#: SDS 6.2 platform/landing safe inset (metres).  Bind from the frozen single
#: source; the fallback keeps a clear, documented dependency on spec.py.
SAFE_INSET = float(getattr(_spec, "SAFE_INSET", 0.042))
#: SDS 6.2 success criterion: body must be at least this far past the platform edge.
PLATFORM_BODY_OFFSET = float(getattr(_spec, "PLATFORM_BODY_OFFSET", 0.80))
#: Geometry constants of the frozen table are asserted below, not re-invented.
_TILE_FOOTPRINT = (24.0, 8.0)
_STAIR_GROUND_HALF_WIDTH = 2.0
_FOOT_HALF_WIDTH = 0.4


def _fail(message):
    raise RuntimeError("v07 terrain: " + message)


# Playback-only override, configured before constructing an environment.
_PLAYBACK_PLATFORM: tuple[int, float, float, int | None] | None = None
_AIR_PLATFORM: tuple[float, float, float] | None = None


def configure_air_platform(*, length: float, width: float, height: float):
    """Replace only the flat playback tile with an isolated elevated slab."""
    global _AIR_PLATFORM
    if _AIR_PLATFORM is not None or _PLAYBACK_PLATFORM is not None or _TABLE_CACHE or _LAST_BUILD is not None:
        raise RuntimeError("Configure playback terrain once before construction")
    if not all(math.isfinite(v) and v > 0 for v in (length, width, height)):
        raise ValueError("Platform dimensions and height must be finite and positive")
    if width > 4 or length < 3:
        raise ValueError("Playback platform requires width <= 4m and length >= 3m")
    _AIR_PLATFORM = (float(length), float(width), float(height))


def configure_playback_platform(*, tile_id: int, length: float, width: float = 0.8,
                                steps: int | None = None):
    """Extend one platform in this process; never edits the frozen training spec.

    Call once, before any geometry/map cache or environment is constructed.
    """
    global _PLAYBACK_PLATFORM
    if _AIR_PLATFORM is not None or _PLAYBACK_PLATFORM is not None or _TABLE_CACHE or _LAST_BUILD is not None:
        raise RuntimeError("Configure playback platform once before building terrain")
    table = build_terrain_specs()
    if not 0 <= tile_id < len(table) or table[tile_id].N == 0:
        raise ValueError("Playback platform requires a stair tile")
    if not math.isfinite(length) or length < 2.0:
        raise ValueError("Playback platform length must be finite and >= 2 metres")
    if not math.isfinite(width) or not 0.8 <= width <= 4.0:
        raise ValueError("Playback platform width must be within [0.8, 4] metres")
    if steps is not None and (type(steps) is not int or steps <= 0):
        raise ValueError("Playback stair steps must be a positive integer")
    _PLAYBACK_PLATFORM = (tile_id, float(length), float(width), steps)


def platform_width(tile):
    """Landing width, independently of the unchanged stair tread width."""
    if _PLAYBACK_PLATFORM is not None and int(tile.tile_id) == _PLAYBACK_PLATFORM[0]:
        return _PLAYBACK_PLATFORM[2]
    return float(tile.width)


# --------------------------------------------------------------- spec binding
def build_terrain_specs() -> tuple:
    """Return the 23-tile v0.7 table and assert its SDS consistency.

    By default the returned tuple is :data:`spec.TERRAIN_TABLE` (same objects).
    An explicit playback override replaces only the selected platform row, so
    geometry, map, contacts, reward height lookup, success and timeout all read
    one ``TerrainSpec`` per tile.
    """
    table = tuple(_spec.TERRAIN_TABLE)
    if len(table) != int(_spec.PHYSICAL_TILE_COUNT):
        _fail(f"expected 23 tiles (flat + T0..T21), got {len(table)}")
    if tuple(int(s.tile_id) for s in table) != tuple(range(_spec.PHYSICAL_TILE_COUNT)):
        _fail("tile ids must be 0..22 in order")
    flat = table[0]
    if int(flat.N) != 0 or float(flat.h) != 0.0:
        _fail("tile 0 must be the flat tile with N=0, h=0")
    if int(_spec.FLAT_TILE_ID) != 0:
        _fail("FLAT_TILE_ID must be 0")
    for s in table:
        tid, n, h, d = int(s.tile_id), int(s.N), float(s.h), float(s.step_depth)
        width, thickness = float(s.width), float(s.tread_thickness)
        platform = float(s.platform_length)
        if n > 0 and width != 0.80:
            _fail(f"tile {tid}: stair width must be 0.80, got {width}")
        if thickness != 0.01:
            _fail(f"tile {tid}: tread thickness must be 0.01, got {thickness}")
        if platform != 2.0:
            _fail(f"tile {tid}: platform length must be 2.0, got {platform}")
        if n < 0:
            _fail(f"tile {tid}: negative step count")
        if n == 0:
            if h != 0.0:
                _fail(f"tile {tid}: flat tile must have h=0")
        else:
            if not (thickness < h <= 0.20 + 1e-12):
                _fail(f"tile {tid}: step height {h} must be (0.01, 0.20]")
            if d not in (0.30, 0.25, 0.22, 0.20):
                _fail(f"tile {tid}: step depth {d} must be 0.30/0.25/0.22/0.20")
        bounds = tuple(float(v) for v in s.valid_bounds_local)
        if len(bounds) != 4:
            _fail(f"tile {tid}: valid_bounds_local must have 4 entries")
        if n == 0:
            if bounds != (-2.0, 22.0, -4.0, 4.0):
                _fail(f"tile {tid}: flat bounds must be (-2,22,-4,4)")
        else:
            if bounds != (-2.0, n * d + platform, -_STAIR_GROUND_HALF_WIDTH,
                          _STAIR_GROUND_HALF_WIDTH):
                _fail(f"tile {tid}: stair bounds must be (-2, N*d+2, -2, 2)")
        origin = tuple(float(v) for v in s.frame_origin_world)
        if len(origin) != 3 or not all(math.isfinite(v) for v in origin):
            _fail(f"tile {tid}: frame_origin_world must be a finite 3-tuple")
    if _PLAYBACK_PLATFORM is not None:
        tid, length, _, steps = _PLAYBACK_PLATFORM
        tile = table[tid]
        lo, _, ylo, yhi = tile.valid_bounds_local
        n = tile.N if steps is None else steps
        extended = replace(tile, N=n, platform_length=length,
                           valid_bounds_local=(lo, n * tile.step_depth + length, ylo, yhi))
        table = table[:tid] + (extended,) + table[tid + 1:]
    if _AIR_PLATFORM is not None:
        length, width, height = _AIR_PLATFORM
        table = (replace(table[0], width=width, frame_origin_world=(0., 0., height),
                         valid_bounds_local=(0., length, -width/2, width/2)),) + table[1:]
    return table


# ------------------------------------------------------------- box geometry
@dataclass(frozen=True)
class Box:
    """One exact collision box in world coordinates."""

    name: str
    step: int
    center: tuple[float, float, float]
    half: tuple[float, float, float]
    top: float

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        cx, cy, _ = self.center
        hx, hy, _ = self.half
        return (cx - hx, cx + hx, cy - hy, cy + hy)

    def contains_xy(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cx, cy, _ = self.center
        hx, hy, _ = self.half
        return ((x >= cx - hx) & (x <= cx + hx) & (y >= cy - hy) & (y <= cy + hy))


def tile_boxes(tile: "object") -> tuple[Box, ...]:
    """Exact world-axis-aligned boxes of one tile (build order: ground first)."""
    ox, oy, oz = (float(v) for v in tile.frame_origin_world)
    n, h, d = int(tile.N), float(tile.h), float(tile.step_depth)
    width = float(tile.width)
    half_t = float(tread_thickness(tile)) / 2.0
    boxes: list[Box] = []
    if n == 0:
        x0, x1, y0, y1 = tile.valid_bounds_local
        boxes.append(Box("flat", 0, (ox+(x0+x1)/2, oy+(y0+y1)/2, oz-.05),
                         ((x1-x0)/2, (y1-y0)/2, .05), oz))
        return tuple(boxes)
    length = n * d + float(tile.platform_length) + 2.0
    boxes.append(Box("ground", 0, (ox - 2.0 + length / 2.0, oy, oz - 0.05),
                     (length / 2.0, _STAIR_GROUND_HALF_WIDTH, 0.05), oz))
    for i in range(1, n + 1):
        boxes.append(Box(
            f"tread_{i}", i,
            (ox + (i - 0.5) * d, oy, oz + i * h - half_t),
            (d / 2.0, width / 2.0, half_t), oz + i * h))
    boxes.append(Box(
        "platform", n,
        (ox + n * d + float(tile.platform_length) / 2.0, oy,
         oz + n * h - half_t),
        (float(tile.platform_length) / 2.0, platform_width(tile) / 2.0, half_t), oz + n * h))
    return tuple(boxes)


def tread_thickness(tile) -> float:
    return float(tile.tread_thickness)


def tile_boxes_by_id(tile_id: int) -> tuple[Box, ...]:
    table = build_terrain_specs()
    return tile_boxes(table[_tile_index(tile_id, table)])


def _tile_index(tile_id, table=None):
    table = table or build_terrain_specs()
    if isinstance(tile_id, torch.Tensor):
        tile_id = int(tile_id)
    value = int(tile_id)
    if not 0 <= value < len(table):
        raise ValueError(f"v07 tile id {value} outside the 23-tile table")
    return value


# ----------------------------------------------------------- generator config
@dataclass(kw_only=True)
class V07AtlasCfg(SubTerrainCfg):
    """One sub-terrain patch that builds the entire 23-tile atlas.

    The generator centres a single patch at the world origin, so the collision
    boxes are placed at their absolute ``frame_origin_world + local`` position
    and the returned spawn origin is the compensating shift (world origin).
    """

    def function(self, difficulty, spec, rng):
        del difficulty, rng
        body = spec.body("terrain")
        shift = np.array((self.size[0] / 2.0, self.size[1] / 2.0, 0.0))
        geometries = []
        for tile in build_terrain_specs():
            for box in tile_boxes(tile):
                color = _color_for(tile, box)
                geom = body.add_geom(
                    name=f"v07_{int(tile.tile_id):02d}_{box.name}",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=np.asarray(box.center, dtype=np.float64) + shift,
                    size=np.asarray(box.half, dtype=np.float64),
                    rgba=color, contype=1, conaffinity=1, condim=3,
                    friction=(0.6, 0.005, 0.0001),
                )
                geometries.append(TerrainGeometry(geom=geom, color=color))
        return TerrainOutput(origin=shift, geometries=geometries)


def _color_for(tile, box):
    if box.name == "flat":
        return (0.30, 0.33, 0.36, 1.0)
    if box.name == "ground":
        return (0.30, 0.33, 0.36, 1.0)
    if box.name == "platform":
        return (0.35, 0.70, 0.45, 1.0)
    return (0.30, 0.55, 0.75, 1.0) if box.step % 2 else (0.42, 0.65, 0.82, 1.0)


def terrain_generator_cfg(seed: int = 42) -> TerrainGeneratorCfg:
    """Generator config used by both :func:`build_terrain` and ``env_cfg``."""
    build_terrain_specs()
    return TerrainGeneratorCfg(
        seed=int(seed), size=_TILE_FOOTPRINT, num_rows=1, num_cols=1,
        curriculum=False, border_width=0.0,
        sub_terrains={"v07_atlas": V07AtlasCfg(proportion=1.0)},
        color_scheme="none",
    )


# ------------------------------------------------------------------ indexing
@dataclass(frozen=True)
class GeometryIndex:
    """Per-model-geom indexing of the real compiled collision geometry.

    ``tile``/``step`` are ``-1`` for every non-terrain geom.  ``bounds`` and
    ``support_bounds`` are the world ``(x0, x1, y0, y1)`` of the box (the support
    rectangle joins the last tread and the platform, exactly like R2).
    """

    tile: np.ndarray
    step: np.ndarray
    bounds: np.ndarray
    top: np.ndarray
    support_bounds: np.ndarray
    geom_ids: tuple


def index_geometry(model, specs=None) -> GeometryIndex:
    """Index the v0.7 collision boxes by measured world coordinates.

    Names are irrelevant (the generator renames every terrain geom to
    ``terrain_<n>``); each terrain geom is matched to the one expected exact box
    and every face is validated.  A missing, extra or displaced box is a hard
    error, so the env can never silently run on v0.2 geometry.
    """
    specs = tuple(specs) if specs is not None else build_terrain_specs()
    expected = []
    for tile in specs:
        for box in tile_boxes(tile):
            expected.append((int(tile.tile_id), box))
    ngeom = int(model.ngeom)
    tile = np.full(ngeom, -1, dtype=np.int64)
    step = np.full(ngeom, -1, dtype=np.int64)
    bounds = np.zeros((ngeom, 4), dtype=np.float64)
    top = np.zeros(ngeom, dtype=np.float64)
    support = np.zeros((ngeom, 4), dtype=np.float64)
    used = np.zeros(len(expected), dtype=bool)
    terrain_body = None
    for b in range(model.nbody):
        if (model.body(b).name or "") == "terrain":
            terrain_body = b
            break
    if terrain_body is None:
        _fail("compiled model has no 'terrain' body")
    ids = [g for g in range(ngeom) if int(model.geom_bodyid[g]) == terrain_body]
    if len(ids) != len(expected):
        _fail(f"terrain geom count {len(ids)} != expected {len(expected)}")
    for g in ids:
        pos = np.asarray(model.geom_pos[g], dtype=np.float64)
        size = np.asarray(model.geom_size[g], dtype=np.float64)
        if not np.isfinite(pos).all() or not np.isfinite(size).all():
            _fail(f"terrain geom {g} has a non-finite position/size")
        best, best_dist = -1, np.inf
        for k, (tid, box) in enumerate(expected):
            if used[k]:
                continue
            dist = float(np.max(np.abs(pos - np.asarray(box.center))))
            if dist < best_dist:
                best, best_dist = k, dist
        if best < 0 or best_dist > 1e-6:
            _fail(f"terrain geom {g} at {pos.tolist()} matches no expected v07 box")
        tid, box = expected[best]
        if not np.allclose(size, np.asarray(box.half), atol=1e-9):
            _fail(f"terrain geom {g} size {size.tolist()} != {box.half}")
        if abs((pos[2] + size[2]) - box.top) > 1e-9:
            _fail(f"terrain geom {g} top face does not match {box.top}")
        used[best] = True
        tile[g], step[g] = tid, int(box.step)
        bounds[g] = np.asarray(box.bounds)
        top[g] = box.top
        support[g] = np.asarray(box.bounds)
    if not used.all():
        missing = [expected[k][1].name for k in np.flatnonzero(~used)]
        _fail(f"unmatched expected boxes: {missing[:8]}")
    for tile_spec in specs:
        tid = int(tile_spec.tile_id)
        n = int(tile_spec.N)
        expected_steps = [0] if n == 0 else [0] + list(range(1, n + 1)) + [n]
        observed = sorted(int(v) for v in step[tile == tid])
        if observed != sorted(expected_steps):
            _fail(f"tile {tid} step metadata {observed} != {expected_steps}")
        if n:
            last = np.flatnonzero((tile == tid) & (step == n))
            if len(last) != 2:
                _fail(f"tile {tid} must have last tread + platform")
            platform = np.asarray(tile_spec.frame_origin_world, dtype=np.float64)
            x0 = platform[0] + (n - 1) * float(tile_spec.step_depth)
            x1 = platform[0] + n * float(tile_spec.step_depth) + float(
                tile_spec.platform_length)
            joined = (x0, x1, platform[1] - _FOOT_HALF_WIDTH, platform[1] + _FOOT_HALF_WIDTH)
            for g in last:
                if platform_width(tile_spec) != float(tile_spec.width) and bounds[g, 0] >= platform[0] + n * float(tile_spec.step_depth) - 1e-6:
                    # The widened landing and last narrow tread form a non-rectangular
                    # union: never invent support beside the last stair tread.
                    support[g] = bounds[g]
                else:
                    support[g] = np.asarray(joined)
    return GeometryIndex(tile, step, bounds, top, support, tuple(int(g) for g in ids))


# ------------------------------------------------------------------ building
@dataclass(frozen=True)
class TerrainBuild:
    """A compiled v0.7 terrain, its config and its per-tile geom handles."""

    seed: int
    generator_cfg: TerrainGeneratorCfg
    generator: TerrainGenerator
    spec: mujoco.MjSpec
    model: mujoco.MjModel
    index: GeometryIndex
    geom_ids: tuple
    specs: tuple

    def tile_geoms(self, tile_id: int) -> list[int]:
        value = _tile_index(tile_id, self.specs)
        return [int(g) for g in self.index.geom_ids if int(self.index.tile[g]) == value]


_LAST_BUILD: TerrainBuild | None = None


def build_terrain(seed: int = 42) -> TerrainBuild:
    """Compile the real 23-tile MuJoCo terrain and return geometry handles."""
    global _LAST_BUILD
    specs = build_terrain_specs()
    cfg = terrain_generator_cfg(seed)
    generator = TerrainGenerator(cfg)
    mj_spec = mujoco.MjSpec()
    generator.compile(mj_spec)
    model = mj_spec.compile()
    index = index_geometry(model, specs)
    build = TerrainBuild(
        seed=int(seed), generator_cfg=cfg, generator=generator, spec=mj_spec,
        model=model, index=index, geom_ids=index.geom_ids, specs=specs)
    _LAST_BUILD = build
    return build


def tile_geoms(tile_id: int) -> list[int]:
    """Geom ids of one tile from the most recent :func:`build_terrain`."""
    if _LAST_BUILD is None:
        _fail("tile_geoms() requires build_terrain() first (no implicit atlas)")
    return _LAST_BUILD.tile_geoms(tile_id)


# -------------------------------------------------------------- surface query
_TABLE_CACHE: dict = {}


def _table_tensors(device):
    """Cached ``(bounds, origins, n, h, d)`` tensors of the frozen table."""
    key = str(device)
    cached = _TABLE_CACHE.get(key)
    if cached is None:
        table = build_terrain_specs()
        bounds = torch.tensor([tuple(float(v) for v in s.valid_bounds_local)
                               for s in table], device=device, dtype=torch.float32)
        origins = torch.tensor([tuple(float(v) for v in s.frame_origin_world)
                                for s in table], device=device, dtype=torch.float32)
        n = torch.tensor([int(s.N) for s in table], device=device, dtype=torch.float32)
        h = torch.tensor([float(s.h) for s in table], device=device, dtype=torch.float32)
        d = torch.tensor([float(s.step_depth) for s in table], device=device,
                         dtype=torch.float32)
        w = torch.tensor([float(s.width) for s in table], device=device,
                         dtype=torch.float32)
        pw = torch.tensor([platform_width(s) for s in table], device=device, dtype=torch.float32)
        cached = (bounds, origins, n, h, d, w, pw)
        _TABLE_CACHE[key] = cached
    return cached


def _broadcast_ids(tile_ids, points):
    ids = torch.as_tensor(tile_ids, device=points.device, dtype=torch.long)
    if ids.numel() and (ids < 0).any() | (ids >= _spec.PHYSICAL_TILE_COUNT).any():
        raise ValueError("v07 tile id outside the 23-tile table")
    while ids.ndim < points.ndim - 1:
        ids = ids.unsqueeze(-1)
    return ids


def query_surface(points_world: torch.Tensor, tile_ids: torch.Tensor):
    """Highest exact surface height for world points on each tile.

    ``points_world[..., :3]`` (z ignored) and ``tile_ids`` broadcast to
    ``points_world.shape[:-1]``.  Returns ``(height, valid)`` where ``valid`` is
    the tile's ``valid_bounds_local`` rectangle (the extent of the approach
    ground box).  Outside the rectangle ``height`` is ``nan``.
    """
    if not torch.is_tensor(points_world):
        points_world = torch.as_tensor(points_world)
    points = points_world.to(torch.float32)
    ids = _broadcast_ids(tile_ids, points)
    bounds_t, origins_t, n_t, h_t, d_t, w_t, pw_t = _table_tensors(points.device)
    origins = origins_t[ids]
    n, h, d, width = n_t[ids], h_t[ids], d_t[ids], w_t[ids]
    x = points[..., 0] - origins[..., 0]
    y = points[..., 1] - origins[..., 1]
    flat = n == 0
    lo, hi, ylo, yhi = bounds_t.unbind(-1)
    lo, hi, ylo, yhi = lo[ids], hi[ids], ylo[ids], yhi[ids]
    valid = (x >= lo) & (x <= hi) & (y >= ylo) & (y <= yhi)
    stair_end = n * d
    on_tread = y.abs() <= torch.where(x >= stair_end, pw_t[ids], width) / 2.0
    height = torch.where(
        x < 0.0, torch.zeros_like(x),
        torch.where(
            x < stair_end,
            torch.minimum(torch.floor(x / d) + 1.0, n) * h,
            n * h))
    # Outside the 0.80 m stair footprint only the approach/platform ground
    # exists (open side edges, no side walls): exact highest surface is z=0.
    height = torch.where(on_tread, height, torch.zeros_like(height))
    height = torch.where(flat, torch.zeros_like(height), height) + origins[..., 2]
    height = torch.where(valid, height, torch.full_like(height, float("nan")))
    return height, valid


def query_surface_height(points_world: torch.Tensor, tile_ids: torch.Tensor) -> torch.Tensor:
    """Highest exact surface height at world points; ``nan`` outside the tile."""
    return query_surface(points_world, tile_ids)[0]


def query_surface_height_np(points_world, tile_ids):
    """NumPy convenience wrapper of :func:`query_surface_height`."""
    height = query_surface_height(torch.as_tensor(np.asarray(points_world, dtype=np.float64)),
                                  torch.as_tensor(np.asarray(tile_ids)))
    return height.detach().cpu().numpy()


# ------------------------------------------------------------ platform bounds
def platform_bounds(tile_id: int) -> tuple[float, float, float, float]:
    """Safe inset rectangle of the top platform in the tile-local frame.

    ``(x0, x1, y0, y1)`` with the SDS 6.2 ``0.042 m`` inset applied on every
    side of the real platform box.  A flat tile has no platform, so its result
    is four ``nan`` values (callers must branch on ``N == 0``; the success test
    never considers a flat tile a platform).
    """
    tile = build_terrain_specs()[_tile_index(tile_id)]
    n, d = int(tile.N), float(tile.step_depth)
    if n == 0:
        return (float("nan"),) * 4
    half = platform_width(tile) / 2.0
    x0 = n * d + SAFE_INSET
    x1 = n * d + float(tile.platform_length) - SAFE_INSET
    y0 = -half + SAFE_INSET
    y1 = half - SAFE_INSET
    return (x0, x1, y0, y1)


def platform_start(tile_id: int) -> float:
    """Tile-local x of the top platform front edge (``N*d``)."""
    tile = build_terrain_specs()[_tile_index(tile_id)]
    return int(tile.N) * float(tile.step_depth)


def expected_height(tile_id: int, local_x: torch.Tensor) -> torch.Tensor:
    """Analytic highest ground/step surface height for local ``x``.

    Same table as :func:`query_surface`, but without the ``y`` axis; used by the
    fall/below-support failure mask and by timeout bookkeeping.
    """
    tile = build_terrain_specs()[_tile_index(tile_id)]
    n, h, d = int(tile.N), float(tile.h), float(tile.step_depth)
    if n == 0:
        return torch.zeros_like(local_x)
    steps = torch.floor(local_x / d) + 1.0
    return torch.minimum(steps.clamp_min(0.0), local_x.new_tensor(float(n))) * h


def timeout_steps(tile_id: int, command) -> torch.Tensor:
    """SDS 6.2 stair timeout: ``ceil((2*(2.2 + N*sqrt(h^2+d^2))/c + 10)/dt)``.

    ``command`` is the executed forward command (``c``); flat tiles keep 1500
    control steps.  The current tile's own step depth is used, never ``0.20``.
    """
    tile = build_terrain_specs()[_tile_index(tile_id)]
    n, h, d = int(tile.N), float(tile.h), float(tile.step_depth)
    c = torch.as_tensor(command)
    if n == 0:
        return torch.full(c.shape, int(getattr(_spec, "FLAT_EPISODE_STEPS", 1500)),
                          dtype=torch.long, device=c.device)
    diagonal = math.sqrt(h * h + d * d)
    seconds = 2.0 * (2.2 + n * diagonal) / c.clamp_min(0.01) + 10.0
    return torch.ceil(seconds / float(_spec.CONTROL_DT)).long()


def tensorize_geometry(index: GeometryIndex, device):
    """Device tensors used by ``contacts``/``events`` (same values as index)."""
    return dict(
        bounds=torch.as_tensor(index.bounds, device=device, dtype=torch.float32),
        top=torch.as_tensor(index.top, device=device, dtype=torch.float32),
        tile=torch.as_tensor(index.tile, device=device, dtype=torch.long),
        step=torch.as_tensor(index.step, device=device, dtype=torch.long),
        support_bounds=torch.as_tensor(index.support_bounds, device=device,
                                       dtype=torch.float32),
    )


def atlas_metadata():
    """Compatibility alias: the v0.7 table (never the v0.2 atlas)."""
    return build_terrain_specs()
