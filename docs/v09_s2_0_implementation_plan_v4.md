# A2 v0.9 S2.0 T2 可执行栈与 V4 修正（v4）

**状态：** 纯 CPU；未训练、未仿真、未 `env.step`、未 Git 写
**Supersedes：** `docs/v09_s2_0_implementation_plan_v3.md`（SHA256 `05edae04c7d798e0ccd7d76185cb9a98ba447a04d8b5120797f55fdec4cfe2ea`）
**裁定：** DSH-Q015～Q026；V4 F-V04-1～6
**实现子 agent：** A2-S2-IMPL

---

## 1. T2 新增文件

| 文件 | SHA256 | 内容 |
|---|---|---|
| `src/tasks/velocity/v09/s2_terrain.py` | `74bc149235d4f2c626d7a64a79c5e7ff39e6e235a561a408d817233c1d049663` | 精确 S2 碰撞盒、解析 `H(base/foot)`、宽窄台、MJCF 编译校验、legacy H 几何 |
| `src/tasks/velocity/v09/s2_state.py` | `544d7932c7568a94eeeff9c256cb55782d3186f644809f65dbdd68bee574e502` | 状态机、`top_arrived` 单次锁存、`D_post` 双零点、步末距离触发、`g_level` |
| `src/tasks/velocity/v09/s2_env.py` | `2684e37a9c9644f67ae71317ca3011369657c4d463c89fe4a7de4f84982fa47c` | 任务适配器、证据记录/落盘、分支配对与前缀证明 |
| `scripts/evaluate_a2_v09_s2_0.py` | `fdf0ef6f31a1efa8a7c222f5a2a3444837ed68eb09c4e4986e463b70ffeea0c8` | 150 分支 + H 6 的 dry-run 执行器 |
| `scripts/rebuild_a2_v09_s2_0_history_projection.py` | `596c4449f2b1a6556266e7e9932bfc2c0ac252e85ec3cd8ab6512c64d7b17225` | 唯一允许写 history 投影的脚本 |

测试：`tests/test_v09_s2_0_terrain.py`、`state.py`、`env.py`、`evaluate_cli.py`，另前述 4 个测试文件。

---

## 2. 地形与状态机

- 碰撞盒：ground、N 个 tread、platform；0.20/0.20/0.80，平台长 20 m、宽 0.8/4 m、顶高 N*h；10/20/40 级；无侧壁/无虚假拼接支撑。
- `query_surface` 返回真实最高表面与边界 valid；窄台外侧只回地面，不伪造平台；宽台 4 m 不被裁窄。
- MJCF 编译与盒子逐项一致（CPU，无 `mj_step`）。
- 状态机：APPROACH→CLIMB→TOP_TRANSITION→POST_WALK→DECEL→HOLD→SUCCESS；S2.0 不停车，DECEL/HOLD 接口在位。
- `top_arrived` 只触发一次；同帧失败优先；距离在步末触发、下一 Actor 才收到新命令；`D_post` 完整登阶用 `top_arrived` 机身 x，平台起步用出生 x。
- `g_level`：真实地形四足同台 + 机身过前缘 0.4 m；0.5 s 保持后线性到 1，条件失效归零并重新计时。**ramp 时长 SDS 未给数值，当前 0.5 s 为命名 provisional 常量，需后续裁定/实测。**

---

## 3. 证据与 CLI

- 每步证据：`s2_metrics` 全部输入 + 逐帧足位/接触/支撑/法向力、contact pairs、PD 目标/实际、joint state、map valid/time、native_sample_id、`c_request/c_exec/a_request/a_applied`。
- `S2EvidenceRecorder.write` 落 NPZ+JSON，创建一次不覆盖；格式与 `s2_metrics` 输入契约一致。
- `branch_execution_plan` 证明 75 对分支共用 case/stream/command/权重/状态；`assert_branches_identical_until` 提供干预前缀逐字段摘要证明。
- CLI dry-run 加载 `S2_DIAG.json` + `S2_DIAG_run_plan.json` + `h_series_case0.json`；非 dry-run 明确拒绝（等待 Q021 预留与 GPU 窗口）。

### dry-run 摘要（`--map-branch both --max-envs 20 --device cpu`）

```text
suite=S2_DIAG; frozen_cases=75; h_series_cases=6
selected_branch=[real, plane_diagnostic]; trajectories=150; branch_pairs=75
entries_by_task={flat_walk:30, stairs_walk:60, platform_walk:60}
env_batches=8; terrain_keys=flat + 40级宽0.8/4.0
physics_started=false
```

---

## 4. V4 修正

| 项 | 修正 |
|---|---|
| F-V04-1 | 混合分支 `exact_clip_denominator` 只取替换支 Σvalid（1080），真实支 NA，不再被真实支稀释 |
| config_identity | 声明与校验统一为 11 项 required pairing 字段 + 7 项 shared identity；缺项报错 |
| missing_reason | `event_not_reached` / `no_valid_refresh` 写入病例与组汇总，含计数；真实支 `exact_clip_missing_reason` 入库 |
| 登记 SHA | register 同时记 `new_receipt_content_sha256`（`5ec1c897…`）与 `new_receipt_file_sha256`（`6f80b6fe…`），反查两者均可定位 |
| 单一状态源 | register 唯一权威；`history_index_v2.json` 改为只读派生投影（含 `read_only_projection`、`register_sha256`、`projection_sha256`），仅重建脚本可写，手工/非派生写入报错 |

当前登记：`supersede_register.jsonl` `426b3ffc…`（2 行：旧格式 + Q026 双 SHA 升级行，幂等按升级行去重）。
当前投影：`history/history_index_v2.json` `1d94f952…`。

---

## 5. 测试

```bash
/home/xxs/anaconda3/envs/a2/bin/python -m pytest -q tests/test_v09_s2_0_*.py
```

**113 passed。** 覆盖：地形边界/坐标/宽窄台/拼接/MJCF 编译；状态机顺序/top 单次/同帧失败/步末距离/timeout/c_exec 链/g_level 保持与归零；证据字段/落盘/缺项/分支配对/前缀证明；CLI dry-run；V4 混合分支、原因码、登记双 SHA、投影不可编辑。

既有 v09 非 `resource_cli` 套件此前 273 passed；`test_v09_resource_cli.py` 21 项 legacy 失败与本任务无关。

---

## 6. 仍无法 CPU 验证 / 待 GPU

1. 实际 `mjwarp`/mjlab 物理推进、接触力、执行器延迟、传感器噪声与 150 条真实轨迹。
2. 两分支“干预前逐帧一致”在真实物理流上的运行时证明（当前有计划和前缀摘要工具，未跑真实序列）。
3. `g_level` ramp 真实时长（SDS 仅给 0.5 s 保持；ramp 0.5 s 为 provisional）。
4. H 六例首足/四足入台归因仍需逐帧足位/接触补采（≤6 条，已计入预算）。
5. 物理/奖励/终止配置 SHA 在真实 env 装配时的绑定。
6. Q021 正式原子预留尚未写入（`q021_reservation_written=false`）；预算仍为 263,888,370 草案。

未执行：训练、仿真、GPU、梯度、Git 写。
