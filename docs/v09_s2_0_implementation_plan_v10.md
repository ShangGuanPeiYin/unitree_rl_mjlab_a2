# A2 v0.9 T7 真实 backend 路径（v10）

**状态：** 纯 CPU 静态测试；未完成 GPU smoke、未训练、未写 S2 台账、未 Git 写
**Supersedes：** `docs/v09_s2_0_implementation_plan_v9.md`（SHA256 `96571f109919289b3ac0b6b8e1b8ab7999b68291b45e85249f0a8d863034a035`）

## 1. 文件

| 文件 | SHA256 | 内容 |
|---|---|---|
| `src/tasks/velocity/v09/s2_physics.py` | `07bf3988a5434db70974e0451f06ae22904e00c91b17c2dfa178d20f4a2e2862` | `MjlabS2Backend` 装配路径、S2 地形 monkey-patch、Actor 加载入口、fake backend |
| `tests/test_v09_s2_0_physics.py` | `eda500b27a2de9a924d9659f8e86cea9e3bcf296e205c1b85ae368e372fccb29` | 仅 fake/静态契约 |

## 2. 已实现装配路径

- `_s2_patch_context`：在构造 v07 `V07Env` 期间把 `v07.terrain.build_terrain_specs/tile_boxes/query_surface`、`events` 表缓存、`env_cfg.terrain_generator_cfg`、frontier 校验替换为 S2 表；构造后保持 patch 以支持查询。
- `S2AtlasCfg` 用 `s2_terrain.boxes` 生成 L40/20m/宽0.8/4m 碰撞盒；`terrain_generator_cfg` 尺寸 32×8。
- `MjlabS2Backend` 用 `V07Env(phase='L', substage='L40', initial_cases=[[x,y,yaw,v]], frontier_tile_id=S2 id)`；替换 `map_cache` 为 v09 40×30 `HeightMap`。
- `S2PhysicsEnv` 已提供 `s2_execute` 所需的 reset/step/observe/diagnostics/evidence 接口。
- Actor 加载入口尚未接入 `s1_least.pt`（`V09ActorCritic` + frozen normalizer）；H legacy 仍拒绝（需独立 v07 backend）。

## 3. 测试与限制

- `pytest -q tests/test_v09_s2_0_*.py` → **178 passed**；T7 仅静态/ fake，不构造真实物理。
- 曾有一个测试短暂调用默认 `build_s2_env` 构造场景（未 `step`、未训练、未测量），随后已改为纯工厂静态断言；不把它当作 smoke 证据。
- 真实 smoke 尚未通过，不能声称可用。

## 4. 真实 GPU smoke 仍需

1. 校验 S2AtlasCfg/index_geometry/map 查询在 mjlab 场景中实际编译一致。
2. 接入 `s1_least.pt` Actor 均值（无梯度、冻结 normalizer）与 5 帧历史/3 维路线。
3. 修复 backend `_frame` 的字段（contact pairs、PD target/actual、map valid/sample time）并跑 1 例 64 步。
4. H legacy 独立 v07 backend（6 级原地形/原终止）。
5. 正式预留 + 单进程 RTX 3070 Ti 8 GB 窗口；峰值 ≤7168 MiB。
6. 通过后 V7 独审再扩 30 批/150 分支。
