# A2 v0.9 T6 S2 物理装配接口（v9）

**状态：** 纯 CPU；未训练、未仿真、未 GPU、未写 S2 台账、未 Git 写
**Supersedes：** `docs/v09_s2_0_implementation_plan_v8.md`（SHA256 `2ab4cf7309fcb612642e390c99f0d4643a5d37fe5a8f124f66f303ec78486403`）

---

## 1. T6 文件

| 文件 | SHA256 | 内容 |
|---|---|---|
| `src/tasks/velocity/v09/s2_physics.py` | `2acd3715c36c5cc0aa8f05be3e858a92b2f0c090ed99c6479c1dd6055c1bf234` | `S2PhysicsEnv` 完整接口、`S2FakeBackend` CPU 桩、`build_s2_env` GPU 前拒绝 |
| `tests/test_v09_s2_0_physics.py` | `122f561f30d432264fb8b9de067e24143695373b0bf569e08a34d864687a27fb` | reset→step→observe、分支前缀、Q016 s−t、H legacy、无 backend 拒绝 |
| `src/tasks/velocity/v09/s2_execute.py` | `37179824…` | 批/预留/commit 驱动 |
| `scripts/run_a2_v09_s2_0.py` | `1da11a48…` | plan-only / smoke 入口 |

## 2. 接口清单（`S2PhysicsEnv`）

- `reset(case_state=None) -> obs`
- `step(action) -> (obs, [done], info)`；终止后拒绝再推进
- `observe() -> {proprio(45), history(5,45), heightmap(3,40,30), route_error(3), c_request, c_exec}`
- `four_feet_on_platform` / `front_foot_top_support` 属性
- `diagnostics() -> {feet_xy, foot_contact, foot_normal_force, contact_pairs, pd_target, joint_position, map_valid, map_sample_time_s, native_sample_id, g_level}`
- `evidence(index) -> trace + map_evidence + state_summary + failure_context + g_level_report`
- `task_adapter`、`recorder`、`machine`、`gate`、`ablation`/`real_evidence`
- `build_s2_env(case, *, device, num_envs, seed, map_branch, h_legacy, backend_factory)`：无 `backend_factory` 时在 `device_report`/GPU 分配前抛 `S2PhysicsAssemblyError`

`S2PhysicsEnv` 负责 S2 状态机、`g_level`、Q016 分支、Q023/Q024 证据、`s2_env.S2EvidenceRecorder`；H legacy 走 `legacy_h=True` + v07 成功/原时限。

## 3. 分支与 Q016

- real：真实 3×40×30 图；plane_diagnostic：`S2PlaneAblation`，完整上楼等待前足真实顶支撑后的首个严格更晚正常刷新，平台/平地首帧；锁存、`s−t`、首替换帧、valid/saturation/exact_clip 分子分母由 ablation summary 落盘。
- 两分支共享 case/spawn/stream/权重/代码；`assert_branches_identical_until` 在 fake 测试中验证干预前逐字段一致。

## 4. H legacy

- `h_legacy=True`：`S2TaskConfig(task='legacy_stairs', legacy_h=True)`，使用 v07 `legacy_v07_success`（支撑且机身≥前缘+0.8 m），原逐例 timeout 1301/1302/1306/1315/1318/1332；不使用 S2 `D_post`。

## 5. 测试

`pytest -q tests/test_v09_s2_0_*.py` → **178 passed**。
T6 新增 5 项 fake 契约；现有 173 项保持通过。未调用真实 mjlab 物理。

## 6. 真实 GPU smoke 仍缺条件

1. **真实 backend_factory**：把 v09/mjlab 的 S2 地形（`s2_terrain` 40 级/20 m/0.8/4 m）、机器人场景、Actor `s1_least.pt` 均值、40×30 地图/10Hz、接触/PD 证据装配进 `s2_physics`；当前 `build_s2_env` 只有 fake backend，缺真实工厂。
2. **H legacy 场景**：v07 6 级原地形与终止条件，需在同一进程外/独立 backend 选择。
3. **GPU 条件**：单进程、RTX 3070 Ti 8 GB、整卡峰值 ≤7168 MiB、主机可用 ≥4 GiB、smoke 1 环境 64 步；运行前打印设备/显存；失败即停保留证据。
4. **正式预留**：`reserve_a2_v09_s2.py --write` 先执行，`q021_reservation_written` 置真；smoke 命令：
```bash
python scripts/run_a2_v09_s2_0.py --execute --smoke --case 0 --steps 64 \
  --device cuda:0 --ledger logs/a2_v09/s2/ledger.jsonl
```
5. **V7 独审**通过后再给窗口；当前无 GPU 执行。

未执行：训练、仿真、env.step、GPU、梯度、Git 写。
