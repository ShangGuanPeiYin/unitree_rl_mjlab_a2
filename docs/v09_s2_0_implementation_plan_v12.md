# A2 v0.9 V7a 四项阻断修复与 smoke 路径（v12）

**状态：** 纯 CPU 测试；正式预留已写入；GPU smoke 待窗口；未训练/未 git 写
**Supersedes：** `docs/v09_s2_0_implementation_plan_v11.md`（SHA256 `e45ed763699107230d2f66d0456b269b2004318f2d0c0d6be052e0b6711d22d2`）

## 1. V7a 修复

| 项 | 落实 |
|---|---|
| tile_id 解析 | `s2_terrain.schedule_tile_id/spec_from_geometry` 按 `(terrain_key,N,platform_width)` 映射到合法 0..6；S2 全部 75 case 通过；表键/ID 唯一；H legacy 仍用原 `tile_id`。 |
| driver/policy | `S2PhysicsEnv.observation()` 别名；driver 单环境向量化 step 兼容；smoke 由 `load_s1_actor` 构造 callable Actor 均值，不传 `policy=None`。 |
| first_step | 改为 `route_x >= 0` 的 APPROACH→CLIMB 转移，与 `top_arrived` 解耦；fake 回归 `enter_climb` 早于 `top_arrived`。 |
| native_sample_id | fake reset 不再预推进；`produced_at_actor_step` 与首次消费步对齐；`S2PhysicsEnv.assert_native_alignment` 强制 `produced ≤ first_consumed`、`s−t≥0`。 |

## 2. smoke 入口

`scripts/run_a2_v09_s2_0.py --execute --smoke --case 0 --steps 64 --device cuda:0 --ledger logs/a2_v09/s2/ledger.jsonl`

顺序：台账存在/预留 → CPU 加载冻结 Actor → 导入真实 backend → 打印设备/显存/case/steps → `ReservationGate` reserve(64) → reset/loop Actor 均值 → 逐例证据 → commit → `smoke_summary.json` + NPZ/JSON。失败也 commit 保留已推进步。H 用 `--h-legacy`。

## 3. 当前状态

- 正式台账：`logs/a2_v09/s2/ledger.jsonl`，run_id `a2-v09-s2-q021-r1`，envelope 263,888,370，plan `773b53c6…`，10 obligations，reserved/spent=0。
- `pytest -q tests/test_v09_s2_0_*.py` → **184 passed**。
- 新增：tile 映射、first_step 顺序、native sample 对齐、Actor 加载/冻结、assembly contract、H backend 分离、patch 恢复。

## 4. 文件/SHA

- `s2_physics.py` `22ff76ba…`
- `s2_terrain.py` `8aed9885…`
- `scripts/run_a2_v09_s2_0.py` `6382bc3c…`
- `tests/test_v09_s2_0_physics.py` `fb2a35b1…`
- 文档本文件。

## 5. GPU smoke 预期首批失败点

1. S2AtlasCfg/index_geometry 对 20 m 平台/4 m 宽行编译匹配。
2. V07Env patched map_cache 10 Hz 与 S2 查询时序。
3. `contacts.top_forces` 索引与 foot_lookup 对 S2 geom。
4. H legacy events 表与 v07 6 级原时限。
5. Actor 输入 dtype/route_3/heightmap 40×30 normalizer。

未执行：训练、仿真、env.step、GPU、梯度、git 写。
