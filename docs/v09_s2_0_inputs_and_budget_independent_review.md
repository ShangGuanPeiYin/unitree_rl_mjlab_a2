# A2 v0.9 S2.0-V1 输入、病例与预算独立核对报告

- 审核 agent：A2-S2-VERIFY（独立验收子 agent，未参与实现）
- 审核时间：2026-11-22
- 仓库/分支/HEAD：`/home/xxs/dog/unitree_rl_mjlab_a2` / `main` / `ab59468cc6e063464b9d3ccdfbe8a90b56d55d09`
- 规范：`doc/sds/SDS_A2_v0.9_Sequential_Map_Walk_Stop_Robustness.md`（v1.2）
- 机器可读件：`logs/a2_v09/s2_0/verification/inputs_budget_audit_20261122.json`
- 范围限制：纯 CPU；**未**做 GPU/物理推进、未启动训练、未改实现/测试/SDS/台账/checkpoint、未做任何 git 写操作。

## 0. 结论分级

| 项 | 结论 | 依据 |
|---|---|---|
| SDS v1.2 身份 | **通过** | 实测 SHA256 与规定一致 |
| S2 入口权重身份（`s1_least.pt`） | **通过** | SHA256 与原件同字节一致 |
| “训练已停止、无 Adam 运行” | **通过** | 进程/GPU/锁/时间戳/台账五路证据 |
| 六例 H 病例独立解析 | **通过**（含 O-1 事实修正） | 六例逐项与冻结生成器完全一致 |
| S2.0 非学习预算上界 | **通过** | 主上界 93,566 控制步 = 0.187% × 0.5 亿 |
| §2.2 病例命名空间（S1 侧） | **通过** | `S1_DIAG` 独立且读取受限于诊断/证据 |
| §2.2 病例命名空间（S2 侧） | **待裁定** | Q-3：S2_DIAG 是否保留 S1 case_key |
| git 工作区差异 | **通过（快照口径）** | 与“无关改动”清单一致；审核期间工作区有并发新增（O-9） |
| S2.0 可执行前置 | **阻断** | B-1（SDS 绑定）、B-2（无 S2 实现/病例/预留）、B-3（并行准入无证据） |
| **总判定** | **阻断**（输入/病例/预算口径本身通过） | 见 §5 |

## 1. 身份核对

### 1.1 SDS

- 工作区 SDS 实测 `SHA256 = 09fa2d4cdd858914c46a51fd52670c20cb20f2b6f3e76e6d266054574f64ccbe`，与规定的 v1.2 SHA 一致（1171 行）。
- 但 **HEAD 版本仍是 v1.1**：`git show HEAD:doc/sds/SDS_A2_v0.9_...md | sha256sum = 500ad32aaa946c927459427f3a65565b15d4b77fca4be214817121d15c742976`。v1.2 仅存在于未提交工作区（见 O-3、B-1）。
- v1.1→v1.2 的 `git diff` 仅 21+/9−，内容是：入口权重条款、S2.0 使用对象、§4.3 两段新增（首米/后段、H 系列复测）、§6.5 报告要求、§9 切换表。**未改动 S1 病例/几何/流定义**。

### 1.2 S2 入口权重（用户指定 `s1_least.pt`）

| 项 | 值 |
|---|---|
| 指定文件 | `logs/a2_v09/s1_seed42_r3/checkpoints/s1_least.pt` |
| 实测 SHA256 | `efa9caa8ce99e2cdff967212063e198b7ae68a6415727083b1db027f615ead24` |
| 与 §1.3 原件关系 | `cmp` **逐字节相同**（原件 `..._latest_000120004608_00009805_f93f026e....pt`，同 SHA，大小 15,583,877 B） |
| 文件名 `f93f026e…` | 是 checkpoint `save_id` 后缀（`00009805_f93f026e…`），**不是**文件 SHA（MD5 亦为 `4ea0747d…`） |
| checkpoint 元数据 | schema 11、stage S1、course S1.2、learning_samples 120,004,608、main 100,024,320、adapt 19,980,288、num_envs 512 |
| 身份裁定 | 不生成 `S1_RESTORED`；首轮 gate_A 仍记未通过（与 §1.3、`logs/codex_changes.md` 一致） |

`codex_changes.md` 记录复制时“未验证副本哈希”；本次由验收方独立复核 `cmp` + SHA256，结论为同字节。

### 1.3 “训练已停止、无 Adam 运行”证据

1. 训练/评估 Python 进程：0（`ps -eo pid,cmd` 仅见 3 个无关 zotero-mcp 进程）。
2. GPU 计算进程：0（`nvidia-smi --query-compute-apps` 结果仅表头）。
3. 锁文件 `logs/a2_v09/s1_seed42_r3/formal_training.lock`：存在但 0 字节、mtime 2026-11-20 18:17，`fuser` 无持有者（exit 1）——**“锁文件存在”不等于在训练**。
4. 时间线：最后 checkpoint 2026-11-22 10:02:40；此后 10:54 为 gate_01 证据目录，11:24–11:34 为用户 H 系列演示（`demo_only`）；台账最后写入 11:33。
5. 台账末尾计数器：`learning_samples = 120,053,760`、`gate_attempts = 1`、`final_suites_used = 0`、`final_recoveries = 0`。
6. 入口权重记录 main = 100,024,320；台账 learning − S1.1 19,980,288 = main 100,073,472，差 **49,152 = 2×(512×48)**，即最后保存后又被强制终止前跑了 2 个完整 rollout。按 §8.3 该 49,152 已计费但**不在入口权重内**（O-4）。

## 2. 六个 H 原始失败病例：独立解析结果

解析对象：`logs/a2_v09/s1_seed42_r3/user_H_series_20261122/` 下 6 个原始 JSON（未读任何实现方中间产物）。六例共性：`case_id=0`、`N=6`、`step_depth=0.30`、`width=0.80`、`platform_length=2.00`、`tread_thickness=0.01`、`tile 1/2/4/7/8/11`、`command vx=0.5, vy=wz=0`、`suite=S1_DIAG`、`stage=S1`、`termination_contract=v07_original_stairs`、`demo_only=true`、JSON 内 `checkpoint_sha256=efa9…`、`sds_sha256=500ad32…`（v1.1，符合 v1.2 头注“历史 S1 证据继续绑定 v1.1”）。

平台区间：楼梯段 `x∈[0,1.8]`，平台 `x∈[1.8,3.8]`（`valid_bounds_local=[-2,3.8,-2,2]`）。下表 `x_term` 取 `episode.forward_distance_end`（绝对局部 x），偏移为相对平台起点 1.8 m。

| 例 | 阶高 m | case_sha256(前12) | 终止步 | 时长 s | x_term | 入台偏移 m | 终止位置 | 终止原因(mask) | max\|y\| | pitch RMS/P99° | JSON SHA256(前12) |
|---|---:|---|---:|---:|---:|---:|---|---|---:|---:|---|
| H00 | 2.00 cm | 84feecc2d400 | 453 | 9.06 | 2.2785 | +0.4785 | **平台** | side | 0.403 | 5.73/17.47 | 35b35d701fbf |
| H01 | 3.08 cm | 9ac0184ff02d | 369 | 7.38 | 2.5711 | +0.7711 | **平台** | nonfoot | 0.081 | 5.42/12.27 | 191d9f545a9c |
| H03 | 5.25 cm | e826d1832915 | 355 | 7.10 | 2.6570 | +0.8570 | **平台** | posture（+side） | 0.409 | 7.16/12.94 | 2a60bed34fb9 |
| H06 | 8.50 cm | ca6621521316 | 343 | 6.86 | 2.5361 | +0.7361 | **平台** | posture（+body） | 0.378 | 10.68/22.68 | b130cd5ece29 |
| H07 | 9.58 cm | 4ad83f92f109 | 193 | 3.86 | 1.0333 | −0.7667 | **楼梯** | nonfoot | 0.085 | 10.47/18.35 | 5a63b2c9bece |
| H10 | 12.83 cm | 9e518aa2735b | 198 | 3.96 | 1.2037 | −0.5963 | **楼梯** | nonfoot | 0.067 | 14.61/27.60 | 61b9aed2b537 |

逐例核对结论：

- 六例原始证据均为 **失败**（`success=false, failure=true, timeout=false, truncated=false`），mask 非空，`terminal_step`/`duration`/`terminal_qpos_sha256` 齐全；六段 mp4 重算 SHA256 与 JSON `video_sha256` **全部一致**；`control_steps` 与 `physical_control_samples` 相等（无子步重复计数）。
- `hard_timeout_steps` 分别为 1301/1302/1306/1315/1318/1332，等于 v07 原上楼硬时限（`original.episode_step_limit(tile_id, 0.5)`），六例均未超时。
- **独立复算冻结病例**：用 `src/tasks/velocity/v09/cases.py` 重建 `build_s1_manifest('S1_DIAG')`，六例 `case` dict（含 `case_sha256`、7 路随机流、spawn 位置/yaw、`hard_timeout_steps`）与 JSON **逐字段完全一致**（manifest SHA `c819d667135580127c11919b7a05bee5ff5d50cacc05719b32013afaa564636d`）。故六例身份可重现、未被实现方改写。
- **原失败位置覆盖（O-1，重要）**：只有 **4/6** 例失败于“进入平台后的正速度巡航”（H00/H01/H03/H06，入台 0.48–0.86 m）；**H07、H10 在楼梯上失败**（x=1.03/1.20，约第 3–4 级），从未到达平台。终止原因分布：side 1、posture 2、nonfoot 3（H03 同时 side、H06 同时 body）。因此 SDS §4.3 的“原故障发生在进入平台后的正速度巡航中”只能作为总体观察，不能逐例套用；“覆盖各自原失败位置”必须按每例自己的终止 x 判定。
- 六例 `metrics.quality_pass` 均为 false，`quality_reasons` 含 pitch_rms/pitch_p99/speed_rmse（H00/H06/H07/H10 还含 pitch_continuous，H01/H06/H07/H10 含 nonfoot_contact）；`platform_support` 四足全 false（无一达到四足平台支撑）。
- **字段不足（O-2）**：JSON 无 `g_level`、`top_arrived`、前足/四足入台时刻、接触几何对/法向力、PD 目标、地图三通道与采样时刻；`frame_audit` 仅隔步记录 pitch/phase。**不满足 §4.3 首米/后段逐例报告要求**，作为“原故障与终止位置”证据可用，完整复测需按原条件配对补采（基线侧计入 S2.0 非学习预算）。

## 3. S2.0 非学习预算独立上界

### 3.1 口径

- §2.5：S2 非学习上限 4 亿；S2.0 子上限 0.5 亿，从 4 亿扣；诊断额度是阶段非学习额度的子上限，不额外相加；每批最坏费用 = **实际批环境数 × 该批最长硬 timeout 步数**；病例数/有效轨迹步数另记，不能冒充物理费用；物理子步另记 4 倍。
- §4.2：`T_max = 2(L_approach + N√(h²+d²) + L_arrival + D_post)/v + T_ramp + T_hold + 10`，步数向上取整；平台起步 `L_approach=N=L_arrival=0`；完整上楼 `L_arrival=1.2 m`；平地硬 timeout 20.02 s = 1001 步。
- 本次上界采用的取值（§4.3 未逐项规定，列入 Q-6 待裁定）：L40 完整上楼 `N=40, h=d=0.20, 梯宽0.80, 台长20 m, 台宽0.8/4.0 m, 顶高8 m, D_post=5 m, L_approach=1.1 m`；平台直接起步 `D_post=5 m`；每组合 **1 例**、按（任务,速度）拼批；H 复测 `D_post=1.2 m`（原 2.0 m 平台扣 0.8 m 登顶判定余量），hard timeout 取 `max(§4.2 公式, 原 v07 硬时限)`。
- 对照矩阵：完整上楼 6 组（2 宽 × 3 速）+ 平台直接起步 6 组 + 普通平地 3 组 = **15 组**；仅诊断用的“平面高度通道替换”分支按同一 15 组**单列**（合同未定义，Q-2）；H 六例配对：基线 6（本次 S2.0）、候选 6（§4.3 计入 S2 工程/评估，不占 S2.0 子上限）。

计算示例（L40，v=0.3）：`2×(1.1+40×√(0.2²+0.2²)+1.2+5)/0.3+10 = 2×18.6137085/0.3+10 = 134.09139 s → ceil(/0.02) = 6705`。
计算示例（H00，v=0.5）：`2×(1.001124+6×√(0.02²+0.30²)+1.2+1.2)/0.5+10 = 30.8205 s → 1542`。

### 3.2 逐批明细（真实地图分支；替换分支数值完全相同）

| 批 | 组（批内 env） | 批 env | 批最长 T_max (s) | 批最长 timeout 步 | 最坏控制步 | 物理子步 |
|---|---|---:|---:|---:|---:|---:|
| B1 | stairs_walk@0.3（w0.8 + w4.0） | 2 | 134.091 | 6705 | 13,410 | 53,640 |
| B2 | stairs_walk@0.5 | 2 | 84.455 | 4223 | 8,446 | 33,784 |
| B3 | stairs_walk@0.6 | 2 | 72.046 | 3603 | 7,206 | 28,824 |
| B4 | platform_walk@0.3 | 2 | 43.333 | 2167 | 4,334 | 17,336 |
| B5 | platform_walk@0.5 | 2 | 30.000 | 1500 | 3,000 | 12,000 |
| B6 | platform_walk@0.6 | 2 | 26.667 | 1334 | 2,668 | 10,672 |
| B7 | flat_walk@0.3 / @0.5 / @0.6 | 1+1+1 | 20.02 | 1001 | 3,003 | 12,012 |
|  | **单分支小计** | 15 | — | — | **42,067** | **168,268** |
|  | 真实地图 + 平面高度替换（×2） | 30 | — | — | **84,134** | **336,536** |
| B8 | H00/H01/H03/H06/H07/H10 @0.5（基线） | 6 | 31.431 | 1572 | 9,432 | 37,728 |
|  | **S2.0 上界合计** | 36 | — | — | **93,566** | **374,264** |

H 六例逐例 timeout（步）：H00 1542、H01 1524、H03 1561、H06 1572、H07 1563、H10 1558；原 v07 值 1301/1302/1306/1315/1318/1332（Q-4）。

### 3.3 上界与阈值

| 诊断深度 | 控制步上界 | 物理子步 | 占 0.5 亿 |
|---|---:|---:|---:|
| 1 例/组（主上界） | 93,566 | 374,264 | **0.187 %** |
| 5 例/组 | 467,830 | 1,871,320 | 0.936 % |
| 20 例/组（S1_DIAG 同深度） | 1,871,320 | 7,485,280 | 3.743 % |

**判定：三种深度均 ≤ 0.5 亿，最坏 3.74%，余量 ≥96%。**（有效轨迹步数因提前失败会更少，仅另记，不用于降低上述最坏费用。）

### 3.4 与 S2 阶段 4 亿额度的预留关系

- S2.0 子上限 0.5 亿 = S2 非学习 4 亿的 **12.5%**；本次上界只占 4 亿的 **0.0234%**，**子上限不是约束**。
- 0.5 亿是“子上限”而非“切块”：S2.0 实际消耗从 4 亿扣，未用诊断额度不转给其他课程/重试/下一阶段（§2.5）。
- 候选侧六例配对复测 6×1572 = 9,432 控制步，按 §4.3 计入 **S2 工程/评估**，不占 S2.0 子上限。
- 指示性（**非权威**，§2.5 要求 manifest 冻结后用真实 timeout/模型次数/padding 重算）：按本报告 timeout 与 §6.1.1/§6.5 组数，40 次门禁上界 ≈1.643 亿（全部按最大课程 S2.3 的 51 组×20 env 保守计），三套终验 ≈0.794 亿，合计 ≈2.437 亿 + S2.0 0.009 亿 ≈ **2.45 亿 < 4 亿**，仍留 ≈1.55 亿给资源探测/工程重跑。该数字不能替代正式预留。
- **当前不存在 S2 manifest / S2 台账 / 阶段原子预留**（`logs/a2_v09/s2_*` 无实现台账）。§2.5 要求全部规定门禁、三套候选及对照终验、探测/诊断上界之和 ≤4 亿后才准入——该预留缺位即 B-2 的一部分。

## 4. §2.2 病例命名空间

- S1 侧成立：`cases.py` 以 `sha256(generator, SDS, 42, stage, suite, role, episode_id, channel)` 派生 7 路独立流；`training`（仅 `train_calibration`）、`gate_A/B`、`final_1..3`、`S1_DIAG` 命名空间分离；读取器对 `S1_DIAG` 强制 `purpose ∈ {diagnostic, evidence, freeze_audit}`，final 集仅 acceptance/freeze 可读（`cases.py:176-187`）。六例 H 即 `S1_DIAG` case0，身份可复算。
- 原始 H 六例**可以**作为 §4.3“固定六例复测”的原故障证据：它们正是 SDS 点名的病例、由入口同一权重 `efa9…` 生成、`case`/流/几何/初始状态与冻结生成器完全一致、失败与终止帧保留、视频 SHA 自洽。用途边界按 §4.3 末段：**只回答复现，不用于选模、不估计总体成功率、不替代 27 组独立验收**；`demo_only=true` 也支持这一限定。
- 但 S2.0 复测的 **suite 命名空间尚未定义**：现有 `case_key` 的 stage 字段为 `S1`（Q-3）。若为 S2 “重新生成”六例，会改变 spawn/流（见 B-1 的量化反例），违反 §4.3“以对应 JSON 初始状态、随机流及几何为准”。

## 5. git 工作区差异核对

- HEAD `ab59468`（`docs: record H series demonstration videos`），分支 `main`。
- 已跟踪修改（`git diff --numstat`）：

| 文件 | +/− | 判定 |
|---|---|---|
| `doc/sds/SDS_A2_v0.9_Sequential_Map_Walk_Stop_Robustness.md` | +21 / −9 | 即 v1.2 内容，与上文一致 |
| `logs/codex_changes.md` | +8 / −0 | H 系列记录 + `s1_least.pt` 复制记录 |
| `scripts/play.py` | +9 / −0 | 仅新增 v02 / `Unitree-A2-IdealHeightMap-Stairs` 路由 |
| `scripts/train.py` | +9 / −0 | 同上 |

- 未跟踪项为**移动快照**（O-9）：首次快照约 13:38 为 185 项，最终快照约 13:58 为 192 项（`doc/` 22、`docs/` 9、`scripts/` 49、`src/` 32、`tests/` 80）。
- 增量 7 项 = 实现方在审核期间（13:53–13:56）新建的 6 个未跟踪 S2 文件 + 本报告：
  `src/tasks/velocity/v09/{s2_spec,s2_cases,s2_metrics}.py`、`scripts/freeze_a2_v09_s2_0_cases.py`、`tests/test_v09_s2_0_{cases,metrics}.py`、`docs/v09_s2_0_inputs_and_budget_independent_review.md`。
  这些**不是**“原有无关改动”，也**未经本 V1 审核**；其中 `s2_spec.py` 采用 S2=v1.2、S1 流盐固定 v1.1 的双身份写法，是对 Q-1 的一个可能答案，须在代码评审轮单独验收。
- 首次快照前，未跟踪内容为 v0.2–v0.8 历史文档、旧版 `src/tasks/velocity/v02..v05_r2`、旧测试与诊断脚本，**未见 v09/S2 实现或 S2.0 证据夹带**。
- 结论：已跟踪修改与“无关未提交改动”清单一致；本次审核未执行任何 git 写操作，原有改动原样保留；并发新增的 6 个 S2 文件不计入本报告的通过结论。

## 6. 阻断项

### B-1（阻断）SDS 双身份未闭合

- 条款：§1.1 / §2.2 / §10.3。
- 事实：`src/tasks/velocity/v09/spec.py:12` 仍为 `SDS_SHA256='500ad32…'(v1.1)`；`spec.validate_sources()`（被 `scripts/video_a2_v09_s1.py:40`、`evaluate_a2_v09_s1.py:36`、`train_a2_v09_s1.py:49`、`prepare/probe/calibrate/compare/finalize` 等**全部既有 S1 v09 CLI 入口**调用）实测抛 `ValueError: sds SHA256 mismatch: 09fa2d4c…`；`stream_id()` 又用该常量做流盐，任何 v09 产物都会被盖 `sds_sha256=500ad32`。
- 风险：现有 S1 证据无法在 v1.2 文件下复跑/复核；若把 v09 S1 常量直接改成 v1.2，六例固定病例被重抽——实测 H00 case0 spawn 从 `(-1.001124, -0.027293) yaw -0.038024` 变为 `(-1.057099, +0.022088) yaw +0.000454`，`case_sha 84feecc2… → 3aa6bdc5…`，直接违反 §4.3。
- 并发事实（O-9）：审核期间实现方新建未跟踪 `src/tasks/velocity/v09/s2_spec.py`，以 `SDS_SHA256=v1.2` + `S1_SDS_SHA256_V11` 固定 S1 流盐，方向与 Q-1 建议一致；但该文件未提交、未评审，本报告**不予签字**，留代码评审轮。
- 处置：规划裁定“文档身份”与“S1 病例流盐”解耦（Q-1）；不得由验收方改代码。

### B-2（阻断）S2 实现/病例/台账/预留缺位

- 条款：§2.2 / §2.5 / §4.1 / §4.3。
- 事实：审核时 `logs/a2_v09/s2_*` 除本审核目录外不存在；无 S2 物理任务/状态机（无 `stairs_walk/platform_walk`，仓库仅 v07 `L40` 的 2.0 m 平台，`valid_bounds` 到 x=10，而 SDS §4.1 要求 20 m 顶平台）；无已冻结的 S2 病例 manifest、无 S2 台账与阶段原子预留。审核期间（13:53–13:56）实现方新增未跟踪 `s2_spec/s2_cases/s2_metrics.py` 与 `freeze_a2_v09_s2_0_cases.py`（`s2_spec` 已含 20 m 平台、三任务与矩阵常量），但未提交、未评审、未执行冻结，本报告不据此放行。
- 风险：§4.3 矩阵与六例配对复测无法执行、冻结、复现；4 亿准入无法核验。
- 处置：完成物理任务/状态机 + 冻结 S2 病例 manifest + 原子预留，再重跑本核对；新增 S2 模块需独立代码评审。

### B-3（阻断，前瞻）并行准入无证据

- 条款：§2.3。
- 事实：无 S2.1 资源探测证据；入口权重元数据 `role_state.num_envs = 512`。用户已裁定“1024 优先、失败退 512”（记为**用户执行裁定**，非 SDS 原文顺序）。
- 要求证据：峰值显存 ≤7168 MiB、主机可用内存 ≥4 GiB、无 OOM/非有限/接触溢出、5 热身 + 50 测量 + 一次恢复检查（每次为完整 48 步 rollout 及 5×4=20 次 Adam）、独立探测副本、探测仿真步计费、评估环境数独立于训练 N。
- **并行数—预算一致性检查项**：1024 → 49,152 样本/轮、S2.1 8 亿 → 16,276 轮（effective_cap 799,997,952）；512 → 24,576 样本/轮、32,552 轮（effective_cap 同为 799,997,952）。S2.2 12 亿：1024 → 24,414 轮、512 → 48,828 轮（均为 1,199,996,928）。**若预算/报告按 1024 报而实测按 512 跑，样本数与轮数不一致，属阻断**；降档必须附资源失败证据。

## 7. 待裁定项（§1.4 格式）

**Q-1｜§1.1/§2.2｜SDS 身份与病例流盐**
- 事实：`spec.SDS_SHA256=v1.1` 同时用于 `validate_sources()` 与病例流盐；改 v1.2 会重抽全部冻结病例。
- 不能确定：S2 产物应以何身份绑定 v1.2，同时保持 S1_DIAG 六例不变。
- 选项：① 新增独立 `S1_STREAM_SALT`，文档身份另设 v1.2；② S1 用例保留旧盐、S2 产物盖 v1.2；③ 以 v1.2 重冻 S1_DIAG（不建议，破坏六例身份）。
- 建议：①，禁止重冻 S1_DIAG。影响：决定所有 v09 证据的 `sds_sha256` 字段。
- 被阻塞：S2.0 采集、任何 v09 CLI 启动。

**Q-2｜§4.3｜“仅诊断用的平面高度通道替换”合同**
- 事实：SDS 只给名称，未定义替换哪个通道、替换值、适用组。
- 不能确定：高度通道置为出生支撑平面/零？valid、age 是否保留？
- 选项：① 高度=出生支撑平面、valid/age 真实；② 高度=0 且 valid=1；③ 仅对楼梯组。
- 建议：①并在全部 15 组执行，作为命名诊断开关，训练/终验禁用。影响：决定单列分支的上界与工程正确性判定。
- 被阻塞：S2.0 替换分支 manifest 与报告口径。

**Q-3｜§2.2/§4.3｜六例在 S2.0 的命名空间**
- 事实：原 `case_key` stage=S1、suite=S1_DIAG；§4.3 要求沿用原初始状态/流/几何；§2.2 要求诊断套独立于训练/门禁/终验。
- 选项：① 新建 `S2_DIAG`，**逐字复制**六例 case dict 并保留 `case_sha256`，另加 operation id；② 直接以 `S1_DIAG`+diagnostic 读取并记 S2 操作号；③ 按 S2 盐重生成（违反 §4.3）。
- 建议：①。影响：决定“固定六例”证据链与 case_key 是否含 S2 stage。
- 被阻塞：S2.0 H 复测 manifest。

**Q-4｜§4.2/§4.3｜H 六例硬 timeout**
- 事实：原 v07 1301–1332 步；§4.2 公式（D_post=1.2）1524–1572 步。
- 选项：① 原 v07 硬时限；② §4.2 公式。
- 建议：预算上界取较大者（公式），报告并列两者。影响：H 配对费用上界（本次 9,432 步）。
- 被阻塞：H 复测 timeout 冻结。

**Q-5｜§4.2/§4.3｜平台起点与两个零点**
- 事实：§4.2“平台起步 L_approach=N=L_arrival=0”，§4.1 “平台前缘后 1 m 出生”，§4.2 D_post 自 `top_arrived`，§4.3 报告零点为“四足入台”。
- 选项：① 1 m 出生偏移独立，D_post 自出生；② D_post 自 0.8 m 机身登顶，报告零点用四足入台。
- 建议：②并显式记录两个零点。影响：platform 组 timeout 与首米/后段指标。
- 被阻塞：platform_walk 病例与报告字段。

**Q-6｜§4.3｜S2.0 诊断深度与展开参数**
- 事实：§4.3 只给轴，未给病例数/批大小/N/顶高/D_post，也未说是否需 10/20 级。
- 选项：1 例/组、5 例/组、20 例/组（均可 ≤0.5 亿，见 §3.3）。
- 建议：执行前冻结一份显式诊断 manifest。影响：精确上界与统计可解释性。
- 被阻塞：S2.0 病例清单冻结。

**Q-7｜§2.5｜S2 阶段原子预留时点**
- 事实：无 S2 manifest/台账/预留；0.5 亿仅 4 亿的 12.5%。
- 选项：① S2.0 前完成；② 仅 S2.1 前。
- 建议：①，并把 S2.0 上界单列。影响：S2 阶段准入。
- 被阻塞：S2 阶段预算准入。

## 8. 未执行项

- 未做任何 GPU/物理推进、未启动训练或梯度更新。
- 未修改实现代码、测试、SDS、`logs/codex_changes.md`、预算台账、checkpoint。
- 未执行任何 git 写操作（add/commit/push/checkout/restore/stash）。
- 未做 S2.0 物理复测/基线补采（需预算预留及 B-1/B-2 裁定）。

## 9. 复现命令（纯 CPU）

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
sha256sum doc/sds/SDS_A2_v0.9_Sequential_Map_Walk_Stop_Robustness.md
sha256sum logs/a2_v09/s1_seed42_r3/checkpoints/s1_least.pt
cmp logs/a2_v09/s1_seed42_r3/checkpoints/s1_least.pt \
    logs/a2_v09/s1_seed42_r3/checkpoints/s1-q010-formal-20261120-r1_latest_000120004608_00009805_f93f026e6df3444aba4bfa5833be38a2.pt
ps -eo pid,cmd | grep -E 'train|a2_v09' | grep -v grep   # 应为空
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
/home/xxs/anaconda3/envs/a2/bin/python -c \
  "import sys;sys.path.insert(0,'.');from src.tasks.velocity.v09 import spec;spec.validate_sources()"
# 预期（当前）：ValueError: sds SHA256 mismatch: 09fa2d4c...
```

机器可读明细（六例完整 case dict、SHA、逐批预算表、git 清单、结论）见
`logs/a2_v09/s2_0/verification/inputs_budget_audit_20261122.json`。
