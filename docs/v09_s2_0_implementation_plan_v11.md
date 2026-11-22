# A2 v0.9 T7 smoke 前置补齐（v11）

**状态：** 纯 CPU 静态验证；未跑真实物理/训练；未写 S2 台账；未 Git 写
**Supersedes：** `docs/v09_s2_0_implementation_plan_v10.md`（SHA256 `200111831adec1e4ec8fa0967993c5bff1b0b10effc2919aa09447149c786ee2`）

## 1. Actor 装配

- `s2_physics.load_s1_actor()`：校验 `s1_least.pt` SHA `efa9caa8…`，加载 `V09ActorCritic` 40×30 网络，`eval()`、`requires_grad_(False)`、冻结 Actor/Critic normalizer；逐键 `torch.equal` 校验 state 与 checkpoint 一致。
- `actor_mean()`：`torch.no_grad()` 确定性均值；CPU 小张量 forward `(2,12)` 已测。
- normalizer 不调用 `update_*`；`normalizer_updates=0`、`frozen=True` 断言。

## 2. Backend 证据补齐

`MjlabS2Backend._frame` 已改为：
- `contact_pairs` 从真实 `contacts.top_forces` 逐足生成（geom/normal force/position/surface/valid）。
- `pd_target`/`a_applied` 取执行器实际 processed actions；joint pos/vel 取真实数据。
- `native_sample_id` 绑定地图真实刷新：`refresh_index`、`sample_sim_time_s`、`produced_at_actor_step`；`refreshed` 由 `map_cache.steps==0` 判定，不再用步数近似。
- `map_valid` 取 40×30 地图 valid 通道；`map_sample_time_s` 取真实仿真时间。
- `front_foot_top_support` 取真实 `current_platform` 支撑，不用特权接触位。

## 3. H legacy backend

- `MjlabHLegacyBackend` 独立于 S2 patch：用未修改 v07 6 级原地形、原 `tile_id`、原 `initial_cases`，仅临时放行 frontier 校验；原逐例 timeout 1301/1302/1306/1315/1318/1332；不进入 S2 `D_post` 成功判据。
- `build_s2_env(h_legacy=True)` 自动选择该 backend；H 只允许 real 图。

## 4. 静态一致性

- `assembly_contract()` 断言控制 50 Hz / 物理 200 Hz / decimation 4 / action scale 0.25 / 12 关节顺序 / 默认站姿 / 40×30 地图 / 0.05 m 网格 / x[-0.8,1.2] y[-0.75,0.75] / 5 步刷新与 v07/v09 一致。
- `_s2_patch_context` 只构造期替换 terrain/events/env_cfg/frontier，退出后恢复原函数；测试验证恢复。
- `S2AtlasCfg` 用 `s2_terrain.boxes`；`index_geometry` 走同一 boxes；未改 S1 文件。

## 5. Smoke 自检顺序

`scripts/run_a2_v09_s2_0.py --execute --smoke`：
1. 台账不存在/无预留 → 直接拒绝（GPU 前）。
2. CPU 加载并冻结 Actor；SHA/state 不符 → 拒绝（GPU 前）。
3. 导入真实 backend；缺失 → 拒绝（GPU 前）。
4. 打印设备、显存、case、steps、estimated_control_steps。
5. 当前在没有 GPU 窗口时明确退出 `smoke preflight OK; GPU execution window not assigned`。

## 6. 测试

`pytest -q tests/test_v09_s2_0_*.py` → **182 passed**。
新增：Actor SHA/state/frozen/forward、assembly contract、patch 恢复、H backend 分离、smoke 缺台账拒绝、`--plan-only` 30 批。

## 7. 真实 smoke 仍差

- 实际 `MjlabS2Backend` 构造后 `_frame` 字段的运行时校验（contact pair 数、map valid、refresh id）。
- Actor 均值接入 `S2ExecutionDriver` 的 smoke 执行体（当前只 preflight）。
- H legacy 真实 6 例 smoke。
- 正式预留 + RTX 3070 Ti 8 GB 单进程窗口；峰值 ≤7168 MiB、主机 ≥4 GiB。
- 预期首批失败点：S2AtlasCfg/index_geometry 在 20 m 平台/4 m 宽行的编译匹配；`V07Env` patched map_cache 与 S2 查询时序；contact/top_forces 索引；H legacy frontier 放行后的 `tile_N/h/d` 表一致。
