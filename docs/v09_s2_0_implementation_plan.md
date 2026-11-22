# A2 v0.9 S2.0 实施缺口、病例冻结与分段指标设计

**状态：** 纯 CPU 实施记录，未运行物理、未训练、未做 Git 写操作
**绑定：** SDS v1.2 `09fa2d4cdd858914c46a51fd52670c20cb20f2b6f3e76e6d266054574f64ccbe`
**S1 历史证据：** 继续绑定 SDS v1.1 `500ad32aaa946c927459427f3a65565b15d4b77fca4be214817121d15c742976`，不改写
**规划裁定：** DSH-Q015～Q021（`doc/sds/reviews/v09_execution_rulings.md`，commit `cb0cd59`）
**实现子 agent：** A2-S2-IMPL

本文件只描述 S2.0 的输入冻结基础、指标纯函数、诊断消融和预算上界；行为结论必须由带物理的 S2.0 运行产生，不能用本文件的单测或预算声明替代。

---

## 1. 身份与入口

### 1.1 文档双身份（Q015）

| 上下文 | 文件 SHA | 使用范围 | 校验入口 |
|---|---|---|---|
| S1（历史） | `500ad32a…` | S1 生成器、流盐、种子、case dict、manifest 的所有参与哈希字段 | 保持 `v09/spec.py` 原值，未修改 |
| S2（实现） | `09fa2d4c…` | 新 S2 原生流、S2 清单、S2 预算、S2 证据 | `s2_spec.validate_sds_context('S2')` |

`s2_spec.validate_sds_context` 只接受精确的 `S1` 或 `S2`；没有同时放行两种 SHA 的入口。当前工作区 SDS 文件是 v1.2，因此 `validate_sds_context('S1')` 会按设计抛错；S1 历史校验仍由 `v09/spec.py` 的常量承担，二者不混用。

S2 原生流盐 = v1.2 + `stage='S2'` + 独立 suite（`S2_0` / `S2_DIAG`）+ 独立 generator `v09-s2-sha256-u53-v1`。六个原 H `case_sha`、spawn/yaw/流保持 v1.1 原值并逐字段保留。

### 1.2 S2 入口权重

| 项目 | 值 |
|---|---|
| 批准路径 | `logs/a2_v09/s1_seed42_r3/checkpoints/s1_least.pt` |
| SHA256 | `efa9caa8ce99e2cdff967212063e198b7ae68a6415727083b1db027f615ead24` |
| 原件 | `.../s1-q010-formal-20261120-r1_latest_000120004608_00009805_f93f026e6df3444aba4bfa5833be38a2.pt` |
| 关系 | 同字节；`s2_spec.validate_identity()` 对两个文件都校验 |
| S1 主段学习样本 | 100,024,320 |
| S1 `gate_A` | 未通过记录保留；不生成、不宣称 `S1_RESTORED` |

---

## 2. 现有能力对照与实施缺口

对照 SDS §4.1/§4.2/§4.3、§5、§6.5、§10。

### 2.1 地形、任务与状态机

| 能力 | 现有实现 | 状态 | 缺口/归属 |
|---|---|---|---|
| S2 地形 0.20/0.20/0.80、平台长 20 m、宽 0.8/4 m、顶高 8 m | 只有 v07 23-tile 表（H/D/L 系列，平台长 2 m） | **缺** | 需 `s2_terrain.py` 编译真实碰撞与解析表面查询；`s2_cases.s2_terrain_rows()` 仅冻结参数契约 |
| 六任务适配 `flat_walk/flat_stop/stairs_walk/stairs_stop/platform_walk/platform_stop` | `V09S1Env` 只接受 `flat_walk/flat_stop/stairs` | **缺** | `s2_env.py` 新适配器，不改 `env.py` |
| 状态机 `APPROACH→CLIMB→TOP_TRANSITION→POST_WALK→DECEL→HOLD→SUCCESS` | v07 有 climb/first-step 诊断，无 S2 顺序状态机 | **缺** | `s2_env.py` + 独立 `s2_state.py`（建议） |
| 单次 `top_arrived`、登顶后不 reset | v09 无 | **缺** | `s2_env.py`；指标已接收 `top_arrived` 锁存事件 |
| `D_post = local_x(t) - local_x(top_arrived)`，带符号净位移 | `s2_metrics` 已实现零点/净位移 | **指标具备，环境缺** | `s2_env.py` 锁存并送入指标 |
| 距离触发在步末、下一 Actor 收到零、100 步 ramp | S1 只有平地固定 400/1000 索引方案 | **缺** | `s2_env.py`；S2.0 不停车，未启用 |
| HOLD 1000 步、末 500 步站立窗 | 现有 S1 平地质量窗 | **缺 S2** | `s2_env.py` / `s2_acceptance.py` |
| §4.2 超时公式（L_arrival=1.2、平台项、T_ramp/T_hold、+10 s） | 无 S2 公式 | **已实现** | `s2_spec.timeout_steps`，有单测 |
| 平地硬 timeout 20.02/28.02/30.02 s | S1 `metrics.flat_schedule` 有同类值 | **S2 常量已冻结** | `s2_spec.flat_schedule/flat_stop_schedule` |
| `c_request/c_exec/a_request/a_applied` 审计 | v07/v09 有部分命令与动作记录 | **缺 S2 全链** | `s2_env.py` + 诊断 |
| `g_level` 0/1、0.5 s 渐进、失效归零、真实地形 | 完全没有 | **缺** | `s2_env.py` + `s2_gate.py`（建议） |
| 抬脚目标：平台/平地 0.075、楼梯 0.22 | v07 `clearance_target(step_h, flat)` 有平地/楼梯分支 | **缺逐足真实表面选择** | `s2_rewards.py` |
| 平台真实边界、宽平台不裁成 0.8 m | v07 固定 width 0.8 | **缺** | `s2_terrain.py` + `s2_env.py` |

### 2.2 奖励、损失与训练

| 能力 | 现有实现 | 状态 | 缺口/归属 |
|---|---|---|---|
| §5.1 总公式（0.02Σw f + 0.4Δx I_cruise +2 +8 −10） | v07 17 项奖励、成功 10/失败 −10 | **缺 S2 配方** | `s2_rewards.py` |
| `g_level×pitch`、水平面超速、稳定站立 +1、减速列权重 | 无 | **缺** | `s2_rewards.py` |
| 停车时关闭前进/腾空奖励并清累计 | 无 | **缺** | `s2_rewards.py` |
| §5.4 平地保持损失 `L_anchor`、`β_flat=0.1`、clean-flat 分母 | `distillation.py` 只有动作/值蒸馏 | **缺** | `s2_loss.py`（建议） |
| S2 PPO：新 Adam、std 解冻 [0.05,1]、KL 调 LR | `v07/ppo.py` 有基础 PPO | **缺 S2 包装** | `s2_runner.py` |
| 1024→512 资源准入 | `resources.py` 是 S1 探测 | **缺 S2** | `s2_resources.py`；顺序已按用户裁定记录在 `s2_spec.S2_ENV_ORDER_RULING` |
| schema 11 S2 检查点字段 | `checkpoint.py` 拒绝 stage≠S1 | **缺** | `s2_checkpoint.py`（新文件，不改 S1） |
| S2 预算台账独立命名空间 | `AppendOnlyLedger` header 绑定 S1 SDS/recipe | **缺** | `s2_budget.py`/`s2_ledger.py`；`STAGE_CAPS['S2']` 可复用 |
| 门禁/终验清单 | 只有 S1 75 组 | **缺** | `s2_cases.py` 后续扩展或 `s2_gate_cases.py` |
| §6.5 27 组验收与质量判据 | `acceptance.py` 只有 S1 配对 | **缺** | `s2_acceptance.py` |
| S2 视频/诊断叠加（阶段、g_level、俯仰、滑移） | `video.py` 是 S1 | **缺** | `s2_diagnostics.py` / `s2_video.py` |

### 2.3 S2.0 诊断与指标

| 能力 | 状态 | 位置 |
|---|---|---|
| 15 组×5 例冻结、两地图分支、清单 SHA | **已完成** | `s2_cases.py` + `logs/a2_v09/s2_0/cases/` |
| 六例 H 外层包装、原 case 不变、定向 operation_id | **已完成** | `s2_cases.py` |
| 前足入台/四足入台/g_level/top/失败/停车事件 | **已完成** | `s2_metrics.case_events` |
| 首米/后段/前足→四足/停车后分段 | **已完成** | `s2_metrics.segment_bounds` |
| 正负俯仰、俯仰角速度、前足支撑/法向力、实际速度/c_exec | **已完成** | `s2_metrics.segment_metrics` |
| 每足步幅/步频/完整落地周期、缺测标记 | **已完成** | `s2_metrics._gait_cycles` |
| 进入/失败/未到达/有效步态样本计数 | **已完成** | `s2_metrics.summarize_group` |
| 平台巡航失败 vs 楼梯失败 vs 接近段失败 | **已完成** | `s2_metrics.failure_context` |
| 原失败位置覆盖判断（未验证标记） | **已完成** | `s2_metrics.original_failure_coverage` |
| Q016 高度通道消融 | **已完成**（接口+激活+统计） | `s2_map_ablation.py` |
| 真实轨迹采集、contact pair、PD 目标/实际、地图有效性与采样时刻 | **缺** | `s2_env.py` + 证据导出 |
| S2.0 执行 CLI / 报告生成 | **缺** | `scripts/evaluate_a2_v09_s2_0.py`（下一项） |

---

## 3. S2.0 病例冻结（Q020）

### 3.1 矩阵

- 15 组 × 固定 `case_id 0..4`，共 75 个冻结 case。
- 平地：3 速 × 20 s 巡航。
- 完整上楼：L40（h=0.20、d=0.20、宽 0.80），登顶后 5 m；两平台宽 0.8/4 m × 三速 = 6 组。
- 平台直接起步：8 m 高平台、台前缘向内 1 m、走 5 m；两宽×三速 = 6 组。
- 平台长 20 m；`vy=wz=0`；不停车、无扰动、无随机化、名义动力学。
- 每例同时服务真实图/平面图两个分支；75 对 = 150 条原生轨迹。
- 每个 case 的流为独立 S2 命名流；`hard_timeout_steps` 按该例实际 spawn x 的 L_approach 计算（Q019 距离定义）。

### 3.2 平台起步与零点（Q019）

- 平台 local_x = 40×0.20 + 1.0 = 9.0 m；机身 z = 8.0 + 0.4 = 8.4 m。
- 普通平地/楼梯 spawn 沿用 L 范围：x∈[-1.1,-0.9]、y∈[-0.05,0.05]、yaw∈[-5°,5°]。
- `D_post` 零点：完整登阶为锁存 `top_arrived` 帧机身 `local_x`；平台起步为出生 `local_x`。
- 首米报告零点：完整登阶为四足投影首次全在平台帧的机身 `local_x`；平台起步为出生点。
- 首米末点即 1.0 m 净位移处；后段从该点起；均为带符号净位移，不是路径长或累计正值。
- 平台起步无 `top_arrived`，不领取 +2。

### 3.3 冻结文件

| 文件 | manifest SHA | 内容 |
|---|---|---|
| `logs/a2_v09/s2_0/cases/S2_0.json` | `5efda5abc35ad714dcf163b1648e2ac4ceac24edc12852889ee6cd3d3739acdf` | 75 case、15 组、两分支 |
| `logs/a2_v09/s2_0/cases/h_series_case0.json` | `22ea4111fddec12dbab1d4d89b26535cc53fdb122dbe661e65ae35b1109f517c` | 六例 S2_DIAG 包装 |
| `logs/a2_v09/s2_0/cases/freeze_receipt.json` | 见文件 | 上述 SHA、入口权重、预算草案 |
| 复算 | `python scripts/freeze_a2_v09_s2_0_cases.py` | 创建一次；不同内容拒绝覆盖 |

*（manifest SHA 以最终重新冻结文件为准；本文件写作后若字段调整必须同步更新。）*

---

## 4. 六个 H 原始失败病例（Q017/Q018）

外层 suite `S2_DIAG`；内层 case dict 逐字段保留原 `S1/S1_DIAG` 内容，`case_sha256` 不重算、不改 stage/suite、不把外层字段写入内层。外层记录来源、原件路径、原 `case_sha256`、独立 `operation_id`。

| lesson | h (m) | 原 v07 timeout | 终止机身 x (m) | 终止段 | 距平台前缘 (m) | 终止时 platform_support |
|---|---:|---:|---:|---|---:|---|
| H00 | 0.020000 | 1301 | 2.278524 | platform | +0.478524 | [F,F,F,F] |
| H01 | 0.030833 | 1302 | 2.571113 | platform | +0.771113 | [F,T,F,F] |
| H03 | 0.052500 | 1306 | 2.657047 | platform | +0.857047 | [F,F,F,F] |
| H06 | 0.085000 | 1315 | 2.536059 | platform | +0.736059 | [F,F,F,F] |
| H07 | 0.095833 | 1318 | 1.033322 | stairs | −0.766678 | [F,F,F,F] |
| H10 | 0.128333 | 1332 | 1.203684 | stairs | −0.596316 | [F,F,F,F] |

- 终止位置取 `episode.forward_distance_end`；`forward_distance_terminal` 为其减 spawn 的带符号距离；`route_final` 不是终止机身位置，未用于判段。
- H00/H01/H03/H06 机身终止于平台，但 `platform_support` 全部为假（H01 仅右前足为真）：指标必须区分“机身过平台”与“四足入台”，不得统一写成平台巡航摔倒。
- H07/H10 机身终止于楼梯（平台前缘前 −0.77/−0.60 m），属楼梯段失败。
- 超时选原 v07 逐例值；六例实际和 7,874，同批 padding 上界 7,992，显式预留 9,432。
- 缺失字段（明确标注、未编造）：原始 qpos/qvel（仅 `frame_audit` SHA）、原始 RNG 状态（仅 `rng_state_sha256_before_sample`）、全速率帧审计（仅子采样帧）。这些不影响 case0 的 spawn/流/几何复算，但限制逐帧原轨迹复用。

`original_failure_coverage` 用重测轨迹的 `max_route_x` 对比原终止 x（默认 0.05 m 容差），并记录 `covered / not_covered` 与 `surface_match`；未到达原位置必须标“未验证”。

---

## 5. 分段指标（Q019，`s2_metrics.py`）

### 5.1 事件

- `front_foot_entry`（= `front_foot_top_support`）：首个前足同时满足“§5.3 顶面支撑”且 XY 在平台矩形内的帧。
- `four_feet_platform`：四足 XY 投影首次全部在平台矩形内的帧（不要求高度/接触）。
- `g_level_positive` / `g_level_full`：`g_level>0` / `>=1` 首帧。
- `top_arrived`、`failure`、`stop_request`、`terminal`：首帧锁存；`terminal` 无显式标记时取末帧。
- 平台矩形为路线坐标下的 `[front_x, front_x+length] × |y|≤width/2`，边界含 `1e-9` 容差。

### 5.2 分段

| 段 | 起点 | 终点 | 说明 |
|---|---|---|---|
| `front_entry_to_four_entry` | 前足入台 | 四足入台或失败/终止 | 单列后足仍在末级时的俯仰变化 |
| `first_meter` | 四足入台（平台为出生） | 净位移 ≥1.0 m 或失败/停车/终止 | 1000 mm 边界为 `>=` |
| `post_first_meter` | 首米末点 | 停车请求/失败/终止 | 首米末点共享边界 |
| `after_stop_request` | 停车请求 | 终止 | 停车后轨迹单独统计 |

失败同帧优先：目标达成与 `failure` 同帧时，`completed=False`、`failed=True`、`end_reason='failure'`。某段起点晚于失败帧时该段 `entered=False`、`not_reached=True`。

### 5.3 指标

- 带符号净位移 = `route_x[end]-route_x[start]`。
- 俯仰：`max`/`min` 有符号（deg）、`abs_max`；不预设“前倾=正”，输出 `sign_convention`。俯仰角速度由显式 `pitch_rate` 或 `diff/dt` 计算；单样本段给 `None` + `single_sample_segment`。
- 前足：`any/all` 顶面支撑比例；法向力 mean/min/max（输入缺失则 `None`）。
- 速度：实际路线速度（显式输入或段内位移/时间）与 `c_exec` mean/min/max。
- 步态：每足落地上升沿之间的完整周期；周期数 <2 判缺测，`stride_median_m/frequency_median_hz=None`，理由 `insufficient_complete_landing_cycles`；组内只对有效步态样本取中位数。
- 组计数：`entered/failed/not_reached/completed/valid_gait_samples`；另有 `failure_on_platform/stair/approach`。
- `D_post` 段值：`d_post_segments_m` 与 `d_post_terminal_m`；首米零点仍为四足入台，互不混用。
- 提前失败保留到终止帧；不截断、不补造缺失路段。
- 与同速平地对照：`compare_gait_to_flat` 先校验速度一致，缺测保留 `available=False`，不填 0。

---

## 6. 高度通道消融（Q016，`s2_map_ablation.py`）

- 只替换**有效格**高度通道；valid/age/坐标/10 Hz/采样帧不变；无效格强制 `0/0/1`。
- 高度值：`clip((z_plane_world - z_base_sample)/1.5, -1, 1)`；记录饱和率与原始 min/max。
- 完整上楼：前足首次真实顶面支撑前用真实图；该事件后**首次正常刷新**起改用真实顶平台平面（z=8 m）；未触发记 `exposed=False` / “未暴露”。
- 平台起步、平地：自首帧用出生支撑平面。
- 开关锁存至终止；记录首次替换帧、有效格比例、替换格数、饱和率。
- 只改策略输入；物理/奖励/命令/统计/终止不变。
- 只用于 15 组 S2.0 诊断；H 复测、训练、门禁、终验一律真实图（H 清单 `provenance.map='real only'`）。

---

## 7. 测试

命令（CPU，无仿真）：

```bash
/home/xxs/anaconda3/envs/a2/bin/python -m pytest -q \
  tests/test_v09_s2_0_cases.py \
  tests/test_v09_s2_0_metrics.py \
  tests/test_v09_s2_0_map_ablation.py \
  tests/test_v09_s2_0_budget.py
```

**结果：50 passed。** 覆盖：

| 文件 | 覆盖 |
|---|---|
| `test_v09_s2_0_cases.py` (12) | 双 SHA 上下文、75 例/15 组/5 例、流独立、确定性、case key/SHA、平台 9.0/8.4 m、L40 实际 approach、超时、H 包装/原 SHA/超时/终止位置/内层不变、地形宽平台、S1 常量未动 |
| `test_v09_s2_0_metrics.py` (23) | 事件边界、平台边界 1 cm、首米精确 1.0/0.999、同帧失败优先、失败早于四足、步态缺测、单样本、俯仰正负、g_level 延迟、停车后独立、平台零点、负净位移、组计数、平地对照缺测、D_post 零点、失败段分类、原位置覆盖、足迹支撑区分、trace 校验 |
| `test_v09_s2_0_map_ablation.py` (8) | 仅高度替换、无效 0/0/1、valid/age 不变、replace mask、stairs 触发+刷新+锁存、未触发未暴露、平台/平地首帧、shape/scale 拒绝 |
| `test_v09_s2_0_budget.py` (7) | 5 例真实分批、H 7874/7992/9432、门禁 36/33/51 组、终验 27 组/3 套、probe 1024/512、S2.0 子额度、40 次门禁、学习 rollout |

已有 v09 全套 CPU 测试回归结果：247 passed、21 failed，失败全部在 `tests/test_v09_resource_cli.py`，与本任务新增文件无关（该文件测试旧资源 CLI/legacy 上下文，改动前即存在）；本报告不宣称其通过。

---

## 8. S2 预算预留草案（Q020/Q021，纯 CPU）

### 8.1 S2.0（0.5 亿子额度）

| 项目 | 控制样本 |
|---|---:|
| 真实图分支：15 批 × 5 例（真实 spawn/实际 max timeout + padding） | 210,145 |
| 平面图分支：同 15 批 | 210,145 |
| 两分支合计（150 条原生轨迹） | 420,290 |
| 六例 H 原 timeout 实际和 | 7,874 |
| 六例同批 padding 上界（6×1332） | 7,992 |
| 六例预留 | 9,432 |
| **S2.0 上界合计** | **429,722** |
| 子额度剩余 | 49,570,278 |

不使用 467,830；430,102 旧算例仅作参考。清单/预算必须从真实 5 例 spawn 重算，不能直接抄旧算例。

### 8.2 门禁（最多 40 次，每次取最坏课程）

| 课程 | 组数 | 例数 | 20 例/组上界 |
|---|---:|---:|---:|
| S2.1 | 36 | 720 | 1,620,000 |
| S2.2 | 33 | 660 | 2,061,980 |
| S2.3 | 51 | 1020 | 4,348,500 |
| 40 次门禁（取 S2.3 最坏） | — | — | **173,940,000** |

门禁分解：S2.1 = 平地 234,180 + 10/20 级复习 766,520 + 平台 400,080 + L40 原里程碑 219,220；S2.2 = 平地 234,180 + L40 walk10 721,240 + 复习 766,520 + 平台 340,040；S2.3 = 平地 234,180 + L40 stop10 1,694,480 + L40 walk10 721,240 + 复习 766,520 + 平台 stop10 932,080。

### 8.3 终验（§6.5，27 组×100 例）

| 项目 | 控制样本 |
|---|---:|
| 候选 2,700 例 | 13,249,500 |
| S1 冻结对照 2,700 例 | 13,249,500 |
| 原 6400 平地 900 例 | 1,170,900 |
| 每套合计 | 27,669,900 |
| 三套 | **83,009,700** |

### 8.4 资源探测（§2.3 全套，1024 优先、失败退 512）

| 项目 | 1024 | 512 |
|---|---:|---:|
| 5 热身 + 50 测量（55×N×48） | 2,703,360 | 1,351,680 |
| 100 环境评估（最长 9,472 步） | 947,200 | 947,200 |
| 1 环境视频 | 9,472 | 9,472 |
| 恢复 rollout（N×48） | 49,152 | 24,576 |
| **合计** | **3,709,184** | **2,332,928** |

### 8.5 全阶段原子预留上界（Q021）

| 项目 | 控制样本 |
|---|---:|
| 资源探测 1024 | 3,709,184 |
| 资源探测 512（失败回退预留） | 2,332,928 |
| S2.0 两分支 | 420,290 |
| 六例 H 缺字段补采 | 9,432 |
| 六例候选复测（3 个候选 × 9,432） | 28,296 |
| 最多 40 次门禁 | 173,940,000 |
| 三套终验 + 冻结对照 | 83,009,700 |
| **合计** | **263,449,830** |
| 4 亿剩余 | 136,550,170 |

学习额度（与上述非学习额度互斥）：

| 课程 | 学习上限 | 1024 完整 rollout | 512 完整 rollout |
|---|---:|---:|---:|
| S2.1 | 8 亿 | 16,276 | 32,552 |
| S2.2 | 12 亿 | 24,414 | 48,828 |
| S2.3 | 12 亿 | 24,414 | 48,828 |
| 合计 | 32 亿 | 65,104 | 130,208 |

S2 学习上限 36 亿，尚余 4 亿未分配（不自动转给其他课程）。每轮样本数：1024×48=49,152；512×48=24,576。

**冲突/保守假设：**
- 40 次门禁按“阶段总次数 40”预留（SDS `GATE_LIMITS['S2']=40`），每次取最坏 S2.3 上界；若裁定为“每课程 40 次”，上界将超过 4 亿，需重新裁定。
- “终验候选复测”按 3 个候选各一次六例 H 复测预留（3×9,432）；若只需 1 次，可释放 18,864。
- 门禁/终验清单尚未冻结，本表是上界；Q021 正式预留前必须用真实 manifest、真实 timeout、真实候选数和 padding 重算。

---

## 9. 裁定落实记录

| 编号 | 落实 |
|---|---|
| Q015 | `s2_spec.SDS_CONTEXT_SHA256` + `validate_sds_context`；S1 常量/流盐/`cases.py` 未改 |
| Q016 | `s2_map_ablation.py`：仅高度有效格、clip 公式、触发/首帧激活、锁存、未暴露、记录统计 |
| Q017 | H 外层 `S2_DIAG`，source/path/case_sha/operation_id；内层 case 原样 |
| Q018 | 原 v07 1301/1302/1306/1315/1318/1332；实际 7,874、同批 7,992、预留 9,432；失败段按终止位置/足迹区分 |
| Q019 | `D_post` 零点=top_arrived 或平台出生；首米零点=四足入台；平台 spawn +1 m；带符号净位移 |
| Q020 | 15 组×5 例；平地 20 s、L40+5 m、8 m 平台+5 m；两宽×三速；两地图分支=150 条 |
| Q021 | `s2_budget.py` 逐项上界 263,449,830 ≤ 4 亿；未写台账；正式预留前重算 |

未决语义项：无。剩余为实现缺口，不是待裁定语义（见 §2/§10）。

---

## 10. 下一步建议（按依赖顺序）

1. `s2_terrain.py`：编译 40 级/20 m 平台/两宽真实地形与解析表面查询；用 `s2_cases.s2_terrain_rows()` 的 origin 逐项校验；加碰撞-地图同源测试。
2. `s2_env.py` + `s2_state.py`：六任务适配、S2 状态机、`g_level`、`D_post`、停车/保持、超时；接入 `s2_metrics` 事件/分段与 `s2_map_ablation`。
3. `s2_rewards.py` / `s2_loss.py`：§5 全奖励、§5.4 平地保持；逐项单测（同帧失败优先、只乘一次 dt、平台 0.075）。
4. `s2_checkpoint.py` + `s2_ledger.py`：schema 11 S2 元数据、独立命名空间、40 次门禁/三套终验计数；不得复用 S1 header。
5. `s2_acceptance.py`：§6.5 27 组质量判据与 6300 条轨迹预算；冻结 S1/6400 对照。
6. `scripts/evaluate_a2_v09_s2_0.py`：只在主对话登记预算与执行窗口后运行；之前保持纯 CPU。
7. 回归：本文件四个测试文件全绿；新增文件不得修改 S1 任何文件。

---

## 11. 未执行项（明确声明）

- 未启动或恢复训练。
- 未运行任何仿真、`env.step`、mjwarp 物理推进或评估 rollout。
- 未使用 GPU、未做梯度更新。
- 未执行 Git add/commit/push/checkout/restore/stash；未修改 SDS、`logs/codex_changes.md`、`scripts/play.py`、`scripts/train.py` 或任何既有 S1 v09 文件。
- 未生成 `S1_RESTORED`；原 `gate_A` 未通过记录保持。
