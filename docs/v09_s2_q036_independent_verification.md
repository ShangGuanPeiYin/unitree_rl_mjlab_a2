# DSH-Q036 独立验收报告（S2.1 训练链路）

- 角色：A2-S2-VERIFY（独立于实现方；未修改生产代码、测试、SDS/裁定、ledger 或训练脚本；未运行 GPU/mjlab/训练）
- 审计日期：2026-11-23（14:21–14:35 CST）
- 依据：
  - 裁定 DSH-Q036：`doc/sds/reviews/v09_execution_rulings.md:344-386`
  - SDS v0.9 §4.2/§5/§5.2/§5.3/§5.4/§6.2/§6.5：`doc/sds/SDS_A2_v0.9_Sequential_Map_Walk_Stop_Robustness.md:462-517,565-706,751-777,824-866`
  - 数值验收设计：`docs/v09_s2_1_numerical_acceptance_plan_v13.md`
- 结论（对 `cd18d44`/14:29 WIP 的时点判定）：**`cd18d44` 不满足 Q036 的 A、top、B 三组要求；14:29 WIP 已大幅接近但有两处可复现缺陷。**
- **最终重审（2026-11-23，`0a0276e`/`56820bb`）：见 §12。实现级结论为 PASS（A/top/B 的核心判据经独立参照与真实 trace 验证）；Run A 已在修复提交 `56820bb` 上重跑并通过（F-1 关闭）。整链签字仍为 CONDITIONAL，仅剩 F-2（探针 reward/at_actor 字段重采）、F-3（门禁/训练窗口统一）与 F-4（D_post 零点）待最终 SHA 复核。**

> 审计对象是移动的。审计开始时 `HEAD=cd18d44`（2026-11-23 14:21:57），工作区已有未提交的 Q036 实现。我在 14:29:26 对 7 个文件做了只读快照并记录 SHA256（`logs/a2_v09/s2/q036_verify/snapshot_1429/SHA256SUMS.txt`）；此后 `s2_physics.py` 又在 14:29:43 变动。**本报告对“HEAD”和“14:29 快照 WIP”分别判定**，所有 file:line 均注明属于哪一份。

## 1. 方法与独立性

1. 只读审计：`git show HEAD:<file>` 取已提交内容；`cp` 取工作区快照；未写入任何生产路径。
2. 独立参照：`quality_reference.py`、`timing_reference.py` 只依赖 numpy/标准库，**不 import** `s2_lifecycle`/`s2_quality`/`s2_rewards`/`s2_train_real`/`s2_physics`，按 SDS 与 Q036 原文实现。期望值先手工推导写入 `quality_cases.json`，再用参照脚本回放核对（不是用生产函数生成 expected）。
3. 证据只取已保存文件：V13 CPU evidence、V13 shadow trace、V13 RunB2 trace、S2.0 raw traces、运行日志。
4. WIP 交叉核对：`crosscheck_wip_quality.py` 只读地 import 工作区 `s2_quality`，把与参照相同的逐帧输入喂进去比较结果（不改生产代码）。

### 1.1 本次生成的独立文件（均在 `logs/a2_v09/s2/q036_verify/`）

| 文件 | 作用 | SHA256 |
|---|---|---|
| `quality_reference.py` | Q036 质量状态机独立 numpy 参照 | `514dd8dd963f…` |
| `quality_cases.json` | 24 个手算用例的输入与期望 | `f64d8c8e6be5…` |
| `quality_cases_result.json` | 参照回放结果（24/24 PASS） | `44f4375e7638…` |
| `timing_reference.py` | 步数/时刻独立推导 | `161861331052…` |
| `timing_cases.json` | first-stop/first-zero/ramp-zero/success/protection 表 | `c3e1bdb24eb5…` |
| `audit_traces.py` / `trace_audit.json` | 真实 trace 逐项审计 | `45ad9a4cb931…` / `aa252cc0fb9a…` |
| `recompute_s2_0_top.py` / `s2_0_top_recompute.json` | S2.0 top 事件按冻结安全登顶规则重算 | `6ac93e1ce322…` / `9060f963a098…` |
| `crosscheck_wip_quality.py` / `wip_quality_crosscheck.json` | WIP `S2Quality` 与参照对拍 | `6b9c398f5992…` / `aebaee1edcb3…` |
| `snapshot_1429/` | 被审计文件快照 + `SHA256SUMS.txt` | 见目录内 |

约束：未提交、未推送、未产生 GPU 消耗、未改 ledger。

## 2. 独立质量参照（交付物 1）

### 2.1 规则实现范围

`quality_reference.py` 实现并文档化了 Q036 A 的全部可判定项：

- 巡航窗：`flat_walk 2–20 s`、`flat_stop 2–8 s`；完整上楼自四足首次同入平台机身位置再净前进 1 m；平台直启自出生 +1 m；至减速请求/不停车终点；空窗或 <50 帧（<1 s）判不合格。
- 不可逆锁存：`|pitch|>15°` 连续 10 帧；§6.5 停滞（51 帧窗、`dx<0.15·c_exec`、高水位 `<0.03 m`、连续 50 帧、`c_exec>0.1`、POST_WALK）；首次 `c_exec=0` 后 150 步内未完成连续 25 帧 §5.3 稳定。
- 停车窗：完整 1000 HOLD；末 500 步均值/峰值速度 ≤0.05/0.10 m/s、相对窗首位移 ≤0.10 m、`|roll|,|pitch|≤10°`、离支撑面高度 ≥0.30 m、四足同时有效支撑比例 ≥90 %。
- 终点显式 `success(+8)` / `quality_fail(−10)`；物理失败、超时同帧优先；未闭合窗口为 pending，不得用前缀统计提前判死。

边界口径（本参照的显式约定，需实现方对齐）：

- 时间轴以“已完成控制区间”计，区间 n 为 1-based，`t=0.02·n` 的行为区间末状态；`t=0` 采样行不计一步（Q036 B2）。
- flat 窗取半开区间：`2.0≤t<20.0`（900 帧）、`2.0≤t<8.0`（300 帧）。
- 楼梯/平台窗取 `[开窗帧, 减速请求帧]`。
- 渐变停车：请求帧后 k=1..100，第 100 个恰为 0；由 B1 的 1500 步终点反推，k=100 的零命令帧是最后一个减速帧，HOLD 为其后 1000 个区间（第 500 帧起零命令共 1001 帧，末 500 窗不受影响）。此 ±1 帧口径应在实现方真实 trace 中固定，见 §5 缺口 T-4。

### 2.2 24 个手算用例（全部由参照回放一致）

| 用例 | 结果 | 终点步 | 窗帧 | 原因 |
|---|---|---:|---:|---|
| flat_walk_pass_baseline | success | 1000 | 900 | – |
| flat_walk_pitch_rms_fail | quality_fail | 1000 | 900 | pitch_rms |
| flat_walk_pitch_p99_fail | quality_fail | 1000 | 900 | pitch_p99 |
| flat_walk_pitch_p99_boundary_pass | success | 1000 | 900 | – |
| flat_walk_pitch_over15_latch | quality_fail | 1000 | 900 | pitch_over15, pitch_p99 |
| flat_walk_pitch_over15_boundary_pass | success | 1000 | 900 | – |
| flat_walk_speed_rmse_fail | quality_fail | 1000 | 900 | speed_rmse |
| flat_walk_missing_window | quality_fail | 1000 | 0 | missing_window |
| flat_stop_instant_pass | success | 1400 | 300 | – |
| flat_stop_ramp_pass | success | 1500 | 300 | – |
| flat_stop_recovery_late | quality_fail | 1400 | 300 | recovery |
| flat_stop_support_ratio_boundary_pass | success | 1400 | 300 | – |
| flat_stop_support_ratio_fail | quality_fail | 1400 | 300 | stand_support |
| flat_stop_peak_speed_fail | quality_fail | 1400 | 300 | stand_peak_speed |
| flat_stop_displacement_fail | quality_fail | 1400 | 300 | stand_displacement |
| flat_stop_roll_fail | quality_fail | 1400 | 300 | stand_tilt |
| flat_stop_height_fail | quality_fail | 1400 | 300 | stand_height |
| flat_stop_hold_pending_incomplete | quality_fail | 1400 | 300 | hold_incomplete |
| stairs_walk_pass_1m_window | success | 600 | 401 | – |
| stairs_walk_window_shorter_than_1s | quality_fail | 210 | 11 | window_short |
| stairs_walk_stall_latch | quality_fail | 500 | 301 | stall |
| platform_direct_pass | success | 500 | 401 | – |
| same_frame_failure_priority | physical_failure | 1000 | 900 | physical_failure |
| same_frame_timeout_priority | timeout_fail | 1000 | 900 | timeout |

关键手算示例（完整推导见 JSON `hand` 字段）：10 帧 16° → RMS `sqrt(10·256/900)=1.686548…`、P99=16、锁存 `pitch_over15`；9 帧 16° → P99=`0.01·16=0.16`，通过；900 帧中 10 帧 12° → RMS `sqrt(1.6)=1.264911…`、P99=12；`flat_stop` 末 500 中 440 帧支撑 → 0.88 失败、450 帧 → 0.90 通过；`first_zero=401` 且稳定支撑 601 才出现 → 625 > 551，锁存 `recovery`。

## 3. 步数与时刻参照（交付物 2）

`timing_cases.json` 的核心表（区间 1-based；`0-based trace step = interval−1`；含 `t=0` 采样行时行数 = 区间数 +1）：

| 计划 | 停车请求 | 首零 | 渐变归零 | 计划达标 | 硬保护 | 总区间 |
|---|---:|---:|---:|---:|---:|---:|
| flat_walk | – | – | – | 1000 (20.00 s) | 1001 | 1000 |
| flat_stop instant | 400 (8.00 s) | 401 (8.02 s) | – | 1400 (28.00 s) | 1401 | 1400 |
| flat_stop ramp_2s | 400 (8.00 s) | 500 (10.00 s) | k=100@500 = 0 | 1500 (30.00 s) | 1501 | 1500 |

- `ramp k=1`：区间 401，`c_exec=0.3·(1−1/100)=0.297`；`k=50`=0.15；`k=100` 精确为 0（`1−100/100=0.0`，`exactly_zero=true`）。
- `t=0` 约定：shadow trace 的 `step 0` 是第 1 个控制区间；日志若额外记录 t=0 行，只是多一行，不得多推进一步。
- **已提交 HEAD 的偏差**：`s2_train_real.py:262-267 @ cd18d44` 传入 `completion_steps=1001/1401/1501`，即把三个硬保护步当成正常达标步，`delta=+1`，三种计划全部在保护步才可能成功（`timing_cases.json.implementation_deltas`）。
- **WIP 快照**：`s2_lifecycle.py:64-67` 改为 1000/1400/1500，`hard` 仅作为 `completion+1` 的备份（`s2_lifecycle.py:187-190`）；`tests/test_v09_s2_lifecycle.py` 新增 `test_flat_stop_counts_1400_instant_and_1500_ramp`、`test_flat_walk_endpoint_is_1000_not_1001`，我在 CPU 上执行该测试文件：**11 passed in 1.44s**。

## 4. top 信号审计（交付物 3）

### 4.1 已提交 HEAD `cd18d44`：FAIL

- 训练：`top_signal = four_feet & body_ok & stairs`，其中 `four_feet` 为四足 XY 落在平台矩形内、`body_ok = local > front+0.4`（`src/tasks/velocity/v09/s2_train_real.py:296-308`）。这是 §5.2 的 0.4 m 门控，不是 Q036 要求的 §4.2 原始安全登顶。
- 原始信号被读取后即丢弃：`original_success = state.success.clone()`（`s2_train_real.py:291`）之后未再使用；v07 的成功在该帧被 `evaluate_without_v07_success` 置零（`src/tasks/velocity/v09/s2_physics.py:605-615`）。真正冻结的安全登顶判据仍在 `v07/events.py:229-255`（`body_x≥x_top+0.8`、`SUCCESS_BODY_PLATFORM_MARGIN=0.8`、0.042 m inset、单足 2 帧、≥2 足当前支撑、50 帧 recent、5 帧连续、`|y|≤0.20`、失败优先），可由 `env.climb.climbed/consecutive_steps/current_support` 读取，但 S2 未使用。
- 评估/物理：`body_ok = local[0] > front+0.4`（`s2_physics.py:795`）、`top_arrived=feet_ok and body_ok`（`s2_physics.py:857`）。s2_physics 与训练共用同一个 0.4 派生 top，**存在训练/评估共因**。
- `g_level` 也以该 0.4 top 为起点（`s2_lifecycle.py:88-93`），因此门控与 top 混同。
- `s2_physics.BackendStep.c_exec/c_request` 取自 `sample_command/command`（`s2_physics.py:844-845`），而 S2 又把 `command` 写成 `c_exec`，二者相差一帧且 `c_request` 不是外部请求——S2.2/S2.3 停车时字段语义会错。

### 4.2 工作区 WIP 快照：已按 Q036 修正

- 在 `evaluate` 抑制前捕获原始 v07 success：`_RawV07Success`（`snapshot_1429/s2_physics.py:541-562`）、`capture.record(result.success)`（`:651`）、`top_arrived=top_raw`（`:904`），并有 `take()` 单帧消费与 H legacy 回退（`:843-853`）。
- 训练改用同一原始信号：`raw_top = self._top_capture.take()`、`top_signal = raw_top & self._s2_top_tasks`（`snapshot_1429/s2_train_real.py:324-330`）；四足入台与 0.4 m 仅作为独立记录的 §5.2 门控（`:331-340`）。
- CPU fake 后端也改为 `front+0.8`（`snapshot_1429/s2_physics.py:202-212`），但它是几何代理、不含支撑/连续确认，只能用于 CPU 测试，不能当真实 top 证据。

### 4.3 S2.0 结论受影响情况（已重算）

用 `recompute_s2_0_top.py` 从 S2.0 原始 trace 重算冻结安全登顶（front+0.8、0.042 inset、2 帧单足、≥2 足当前、50 帧 recent、`|y|≤0.20`、5 帧连续、失败优先），覆盖 `stairs_walk/real/N40` 全部 30 例：

| 组 | 报告 top | 重算安全登顶 | 重算区间 |
|---|---:|---:|---|
| w0.8（15 例） | 15/15 | **15/15** | 安全登顶比报告事件晚 **16–20 帧**（同一采样帧时基；Q037 §3 更正，旧“15–19”为 1-based 报告步与 0-based trace 步混算）；登顶→失败窗 **25–38 帧**，全低头比例仍 1.0 |
| w4.0（15 例） | 0/15 | 0/15 | 两种定义都不登顶，结论不受影响 |

代表例（case_id 0，v=0.3, w0.8，以下均为 trace 采样帧 s）：四足入台 1558、0.4 代理首真/报告事件 1560（报告约定 r=1561）、重算安全登顶 1578、失败 1608、末有效物理帧 1607；报告“top 窗 43–55 帧”对应的是 0.4 门控→失败（1608−1558=50），按真实安全登顶只有 30 帧。

**判定**：`v09_s2_0_diagnostic_final.md` §3 的 top 时刻与 top 窗帧数被旧代理污染，**必须改写为“0.4 m 门控窗”或采用本次重算值**；“15/15 登顶后前倾、全低头”的定性结论仍成立（重算 15/15 安全登顶、nose-down 1.0）。原始 trace 覆盖该事件，故属“可重算”而非“未验证”；本文给出可复现脚本与结果。

## 5. 真实 trace 逐项审计（交付物 4）

### 5.1 V13 shadow（`rollout_000000/1.jsonl`，512×48×2）

| 检查 | 结果 |
|---|---|
| `(env_id, step)` 唯一/对齐 | 24576 行/轮，唯一对 24576，**order = step·512+env 全部一致**（time-major flatten 正确） |
| `ret = adv + value` | max |误差| `4.77e-7`（float32 级），两轮均通过 |
| done 分类 | `terminated_mask==done`、`truncated=0`；failure 类 `fall/nonfoot/posture/side` 逐例记录 |
| 跨 rollout episode 连续 | 486/512 连续；26 个 env 在 rollout 边界仍处于终止帧 |
| c_request / c_exec | c_request ∈ {0.3,0.5,0.6}，**所有帧 c_exec==c_request**（S2.1 前 96 步无停车，符合预期） |
| reset 隔离 | 每帧 `done` 的下一条 `g_level=0`、phase 回 APPROACH；但见下 |

**版本问题（关键）**：该 trace 生成于 14:09，早于 `62ea888`（14:16:58），其 tile 映射是旧代码。env 410–435（512 版全部 `platform_walk` N=10 行）**每一帧都失败**（class `fall`，`episode_length` 恒为 0，reward≈−10.08）：旧 `specs[terrain_key]` 映射把 pw0.8/pw4 全部落到 **N=20 tile**（现 `s2_terrain.terrain_specs()` tile 1/3/5 共享 `..._pw0.8`，最后命中 N=20），N=10 平台出生 z=2.4 落在 4 m 高几何内 → 立即 fall。同一 N=10 平台行在 14:18 的 RunB2（62ea888 之后）不再瞬倒，佐证因果。因此：**对齐/ret/GAE/命令恒定性可引用；reset 隔离与逐 env 持久性不能引用这份旧 trace**，必须用当前 SHA 重采。

### 5.2 RunB2（`v13_runB2/{flat,stairs,platform}/smoke_trace.jsonl`）

| 运行 | 结果 |
|---|---|
| flat（env0/1，907 步） | **崩溃**：`ValueError: executed command changed inside an episode`（`v07/env.py:163`，日志 `v13_runB2_flat.log`）；trace 止于 step 906，无 summary |
| stairs（env4/5/6，906 步） | **崩溃**，同上；top 行 46/95/51，POST_WALK 出现后被截断 |
| platform（env7，900 步） | 完成；top 0（platform 任务无 +2，符合设计）；14 次 physical failure 后自动 reset 回 spawn x=3.0 |

**命令链反例（最重要）**：flat env1 为 `flat_stop`（ramp）。trace 显示 step 399 起 `phase=DECEL`、step 499 起 `phase=HOLD`，但 **`c_exec` 从 399 到 905 一直是 0.3 m/s**（507 帧 phase 与命令矛盾），直到 step 906 `local_x=5.0047 m` 才变 0；随后 step 907 触发上面的 stall 崩溃。即：phase（S2Lifecycle）在 8 s/400 区间按规则进入 HOLD，而 Actor 实际消费的命令来自 `_s2_reward` 里第二套“距离 ≥5 m 才停车”的状态机（HEAD `s2_train_real.py:405-423,496`）。这正是 Q036 A1/B3 禁止的双状态机，被真实 trace 直接证明。

**该运行早于修复**：崩溃点由 `2b0508a`（14:19:37，晚于运行）才修复；两份 trace 都没有 `episode_length`、`reward`、`advantage/ret` 字段，`done` 行的 `local_x` 已是 reset 后 spawn 值（3.0），**终止位置不可从该 trace 读取**。因此 RunB2 只能作为“缺陷证据”，不能作为通过证据。

## 6. Q036 逐项判定

图例：PASS / PARTIAL / FAIL / NOT-COVERED。`HEAD`=已提交 `cd18d44`；`WIP`=14:29 快照。

| Q036 项 | HEAD | WIP | 依据 |
|---|---|---|---|
| A1 统一判据、删 `quality_ok` 恒真、逐 env 持久 | FAIL | PARTIAL | HEAD `s2_train_real.py:309` 恒真；WIP 接入 `s2_quality`（`s2_train_real.py:277-300,341`）但门禁/终验仍用另一套 `s2_acceptance.quality_reasons`（`s2_acceptance.py:18-57`）与 `metrics.py:170-200`，未统一；无 `s2_quality` 测试 |
| A2 巡航窗（时间/1 m/空窗/<1 s） | FAIL | PARTIAL | WIP 窗实现于 `s2_quality.py:391-412`，但**窗采样未要求 `c_exec>0`**：交叉核对中“全程零命令”的 flat_walk 被判 success（`wip_quality_crosscheck.json`），违反“空窗不能成功” |
| A3 不可逆锁存（pitch/stall/recovery） | FAIL | PARTIAL | WIP `s2_quality.py:377-381,467-471,693-721`；HEAD 无质量锁存，v07 stall 仅诊断且 `enabled` 恒真（`v07/env.py:766-769`），语义与 §6.5 相反 |
| A4 物理失败优先、合法平台边界分开 | PARTIAL | PARTIAL | 失败优先级两版都有；**合法平台边界（离开有效平台记失败而非物理摔倒）未实现** |
| A5 窗闭合指标 RMS/P99/RMSE | FAIL | PASS | WIP `s2_quality.py:601-628`；交叉核对 3 个指标用例一致 |
| A6 停车窗（1000 HOLD/末 500/位移/姿态/高度/支撑比） | FAIL | PARTIAL | WIP 指标齐全，但 HOLD 样本按**计划区间**计数（`s2_quality.py:473-476`），不按实际 `c_exec==0`；交叉核对中“零命令晚 100 帧”仍判 success |
| A7 终点显式 +8/−10 | FAIL | PASS | WIP `s2_lifecycle.py:181-190` + `s2_quality.py:536-546`；`s2_rewards.py:226-233` 付 +8/−10，失败优先 |
| A8 同帧失败优先、bootstrap=0 | PARTIAL | PARTIAL | 奖励与终止优先级有；GAE mask 未新证据，需当前 SHA 的真实 trace |
| A9 组级相对/§5.4/L40 例外 | NOT-COVERED | NOT-COVERED | 本报告不含组级门禁/锚定数值；L40 的 top+2/success+8 由奖励路径支付但未用真实样本验证 |
| top-1 0.4 门控不是 top | FAIL | PASS | HEAD `s2_train_real.py:306-308`；WIP `s2_train_real.py:324-330` |
| top-2 原始安全登顶（0.8/支撑/连续） | FAIL | PASS | HEAD `s2_physics.py:795,857`；WIP `s2_physics.py:541-565,651,904`（不可改的冻结判据在 `v07/events.py:229-255`） |
| top-3 一次 +2、不 reset、L40 例外 | PARTIAL | PARTIAL | `top_event` 单次锁存；真实样本未覆盖 |
| top-4 记录四足首帧/g_level/top/§6.5 起点 | FAIL | PARTIAL | WIP 记录 `entry_step/cruise_start_step`（`s2_quality.py:383-398`），`g_level` 仍由 lifecycle 的 top 派生 |
| top-5 训练/评估共因独审 | FAIL | PASS(代码) | HEAD 共因；WIP 两侧都改为原始信号；**但无真实 GPU 验证** |
| B1 1000/1400/1500 | FAIL | PASS | HEAD `s2_train_real.py:262-267`；WIP `s2_lifecycle.py:64-67` + 11 个生命周期测试通过 |
| B2 1001/1401/1501 仅保护 | FAIL | PASS | WIP `s2_lifecycle.py:187-190` |
| B3 400 后请求、k=1..100 第 100 为 0 | FAIL | PASS | HEAD 的 flat_stop 在奖励路径按 5 m 触发（`s2_train_real.py:405`）；WIP `s2_lifecycle.py:135-152`，测试断言 k100=0；`c_request→0` 见 T-5 |
| B4 按实际命令记录首零/HOLD/末 500 | FAIL | PARTIAL | WIP 记录 `first_zero_step`，但 HOLD 用计划区间（A6 同因） |
| B5 不得拖到保护超时 | FAIL | PASS | WIP 正常终点即成功/质量失败，保护仅备份 |
| C1 对最终 SHA 重跑受影响验证 | FAIL | FAIL | V13 CPU evidence 生成于 `2026-11-23T06:17:26Z`（=14:17:26 CST，晚于 62ea888 28 s），**不覆盖** 2b0508a、cd18d44 与 WIP |
| C2 真实训练入口+逐帧参照签字 | FAIL | FAIL | 见 §8/§9 |
| C3 真实未达状态须补可控边界且区分 | FAIL | PARTIAL | 参照已给出边界用例；实现方尚无真实/回放边界证据 |
| C4 Q035+签字后从 `s1_least.pt` 起 r2 | 未达 | 未达 | 前置未完成 |

## 7. 现有 V13 CPU 证据与真实 trace 各自闭合了什么

- **V13 CPU evidence（`logs/a2_v09/s2/v13/v13_cpu_evidence.json`，`all_passed=true`）闭合**：13 个手算奖励用例、g_level 序列、PPO 存储 `(env_id,step)` 对齐、GAE 边界（独立复算）、异常结算、r1 拒绝链、§5.4 锚定手算。它**明确自述** `blocked_real_backend`（Run A–E）与 `not_verified`（无真实 rollout、无锚定梯度、无逐 env mask、无动作链样本）。
- 它**不覆盖**：Q036 质量状态机、top 定义、步数/时刻、逐 env 质量持久、停机首零/HOLD、S2.0 top 窗。
- **shadow trace 闭合**：time-major flatten、`(env_id,step)` 对齐、`ret=adv+value`、c_request/c_exec 恒定、terminated/truncated 分离、跨 rollout episode 连续（486/512）。**不闭合**当前树的 reset 隔离/持久性（26 env 旧代码伪影）。
- **RunB2 trace 不闭合任何通过项**：两份运行崩溃、schema 缺字段；仅提供“双状态机/命令不同源”的反例。
- 只有**旧代码版本**（62ea888 之前）有真实 shadow；RunB2 是 62ea888 之后但修复前的版本；**当前 HEAD/WIP 没有任何真实 trace**。

## 8. 签字缺口清单（交付物 5）

### 8.1 可由 CPU 验证闭合（实现方先做）

| 编号 | 缺口 | 位置 | 闭合证据 |
|---|---|---|---|
| T-1 | 质量窗未按 `c_exec>0` 采样，零命令回合可判 success | WIP `s2_quality.py:391-412` | 修后加反例单测；本报告 `wip_quality_crosscheck.json` 已复现 |
| T-2 | HOLD 按计划区间而非实际零命令计数，晚零仍 success | WIP `s2_quality.py:473-476,604-635` | 以 `first_zero_step` 锚定 HOLD，加晚零反例单测 |
| T-3 | 训练与门禁/终验仍是两套质量实现 | `s2_acceptance.py:18-58`、`metrics.py:170-200` vs `s2_quality.py` | 门禁/终验改调同一 `S2Quality`（或反之）并给同一输入的对照测试 |
| T-4 | 渐变 HOLD 起点 ±1 帧口径未固定 | `s2_lifecycle.py:139-154`、`s2_quality.py:443-455` | 用真实/回放 stop trace 固定 first_zero/hold_start/hold_end 三列并写入合同 |
| T-5 | `c_request` 语义冲突：v13 §4.1 要求保持计划速度，WIP 与测试令其停车后为 0 | WIP `s2_lifecycle.py:135`、`tests/test_v09_s2_lifecycle.py` `assert np.allclose(s.c_request[0], 0.0)` | 由裁定方二选一并改合同/验收判据；两版 trace 均无停车 c_request 样本 |
| T-6 | `|pitch|>15` 锁存作用于全部帧而非巡航窗 | WIP `s2_quality.py:377-381` | 明确作用域；若保持全帧，需证明爬楼/接近段不会误锁 |
| T-7 | `s2_quality` 无任何测试文件 | `tests/`（仅 lifecycle 测试改动） | 24 个参照用例的同序移植 + reset 隔离 + pending/closed + 同帧优先级 |
| T-8 | quality 消费 `stop_requested/c_exec` 存在一帧耦合（靠 `request_step=step-1` 补偿） | WIP `s2_train_real.py:414-440`、`s2_quality.py:436-447` | 在真实/回放链路上固定更新顺序，或改为直接由 lifecycle 输出本帧请求 |
| T-9 | 离开有效平台未单独分类为任务失败（A4） | 全链 | 增加平台边界判据并给边界用例 |
| T-10 | 组级/§5.4/L40 例外未复核 | `s2_acceptance.py` | 本报告范围外，需另一份门禁验收 |

### 8.2 必须真实 GPU run（CPU 不可替代）

| 编号 | 缺口 | 闭合证据 |
|---|---|---|
| G-1 | 最终 SHA 的 V13 Run A：两连续 sampling block、512 env、`(env_id,step)`、reset 隔离、质量字段 | 新 trace（含 `episode_length/quality_*/hold_*/c_exec_consumed`）与独立复算 |
| G-2 | 真实 instant 与 ramp 各一条停车序列：first_zero/hold_start/末 500/终点 1400/1500，`c_exec` 单调、`a_applied==pd_target` | 逐帧 trace + 本参照对照 |
| G-3 | 真实安全 top 事件：0.4 门控已开而 top 未开、top 距离零点、L40 同帧 +2/+8 | 真实样本或可控回放（须标注来源） |
| G-4 | top 修复后的 S2.0/评估侧一致性 | 重采受影响病例或接受 §4.3 重算 |
| G-5 | 资源/预算/512 资源复核 | Run E |

## 9. 可签与不可签

**可以签（本报告结论）**

1. 已提交 `cd18d44` 不满足 Q036 A/top/B，不得作为 S2.1 通过证据；其 V13 CPU evidence 只覆盖 62ea888 时点。
2. 两套独立参照（质量 24/24、时刻表）已建立，可作为后续验收判据。
3. S2.0 top 结论必须按 §4.3 重算；w0.8 的“15/15 安全登顶、全低头”仍成立，但 top 时刻晚 16–20 帧、窗长 25–38 帧而非 43–55（Q037 §3 精确重算）。
4. shadow trace 的 PPO 对齐/GAE/命令恒定性可引用；reset/持久性不可引用（旧 tile 映射伪影）。

**不能签**

- 不能签“S2.1 训练链路满足 Q036”：HEAD 明确失败；WIP 未提交、无质量测试、§8.1 的 T-1/T-2/T-3 已被独立复现或代码确证。
- 不能签“真实质量链通过”：现有真实 trace 均为修复前版本，且 flat/stairs 运行崩溃。

**签字前需要实现方提供**

1. 最终提交 SHA（并告知 `s2_physics.py` 在 14:29:43 的后续改动）。
2. `s2_quality` 的 CPU 测试（覆盖本报告 24 用例 + 同帧优先级 + pending/closed + reset 隔离），以及修后 T-1/T-2 的反例测试。
3. 门禁/终验改用同一判据的证据。
4. 真实 instant/ramp 停车 trace 与安全 top trace（G-2/G-3），字段满足 §5.1/§5.2 清单。
5. 裁定方对 T-5（c_request 语义）与 T-6（pitch 锁存作用域）的明确口径。
6. 受影响 S2.0 结论的改写或接受本报告重算结果。

## 10. 限制与未验证项

- 未运行 GPU/mjlab/训练；所有真实行为结论来自已保存 trace 与其日志，未重新测量。
- WIP 仍在变动：快照后 `s2_physics.py`（14:29:43，top 捕获的 H legacy 回退）与 `s2_train_real.py`（14:32:50，仅补 `import torch`）又变；`s2_quality.py`/`s2_lifecycle.py` 未变（`f3ea8e61…`/`cc11e05b…`），§5 交叉核对结论仍适用。本报告对 WIP 的判定以 `snapshot_1429` SHA256 为界。
- 交叉核对只覆盖 flat 非停车/停车路径与楼梯停滞的参照用例；楼梯/平台距离窗、L40 例外、逐 env reset 的 WIP 交叉核对未执行（参照侧已给出，但 WIP 适配需要距离停车 wiring）。
- S2.0 重算用报告中的原始 trace 与几何；未覆盖 plane 分支（plane 全 0 top，结论不受影响）。
- `s2_physics` 的 `_s2_foot_support` 几何边界（tread 半宽 0.4 m、ground 范围）与 §5.3 真实支撑面的一致性未做逐点核验，属代码审读。

## 11. DSH-Q037 §3 follow-up：S2.0 top 重算的等价性证明与更正

裁定 DSH-Q037 §3 要求对离线 S2.0 修订给出“冻结 v0.7 逐足支撑语义”的等价性证明、修正汇总错误、统一时基，并把派生件版本化。执行结果如下（独立完成，未重跑 156 例、未用 GPU、未改生产代码）。

### 11.1 逐足精确重建（取代 v1 代理）

v1（`logs/a2_v09/s2/q036_verify/recompute_s2_0_top.py`）用 `foot_on_platform` + 聚合 `foot_normal_force`，且 `feet_safe` 只在帧级使用。v2（`logs/a2_v09/s2_0/verification/q037_top_recompute_v2/recompute_s2_0_top_v2.py`）按冻结 `PlatformSupportState`（`v07/events.py:229-255,257-268`）重建：

- 逐足候选 = **平台 geom** 顶面法向力 >5 N ∧ 该足在 0.042 m 内缩平台内（`contact_pairs` 的 `geom` 名是全局 `terrain_<id>`；N40 tile 的 box 顺序 `ground,tread_1..40,platform`，故 `platform_id = min(terrain_id) + N + 1`；窄台 offset=1→platform=42，宽台 offset=43→platform=84）；
- 逐足 2 帧支撑、同 surface/tile 持久（单 tile 单平台 geom 下等价于连续 valid）、recent-50；
- 帧级 `body_x ≥ front+0.8`、`|body_y| ≤ 0.20`、四足 safe、≥2 足当前支撑、无失败优先；
- 5 帧连续锁存；**同帧失败优先**（失败帧不产生事件）；
- `feet_xy` 世界→local 减 `frame_origin_world`；`base_xy` 已是 local（v1 多减 spawn y 的写法改正，本例 spawn y=0，未改变数值）。

### 11.2 等价性结论

- 窄台 15 例：v2 精确事件步与 v1 **逐一相同**（`proven_equivalent` 15/15），故 v1 的 **25–38 帧** 结论不是未验证代理，**被证明未变**。
- 等价性检查非平凡：15 例中共 **84 帧**逐足候选在“严格 vs 代理”下不同（例 case 0 的 s=1532–1561 有 6 帧），事件步仍不变。
- 宽台 15 例：两种方法均无事件，且机身最大 x=8.1576 m < `front+0.8=8.8 m`，属**几何证明**（`proven_no_summit_by_geometry` 15/15），与接触/力语义无关。
- 无 `proxy_reconstruction_unverified` 病例；但本证明仅覆盖 30 个 `stairs_walk/real/N40` 病例，plane/flat/platform/H 系列不在此范围。

### 11.3 时基更正（推翻本文 §4.3 的“15–19”）

统一到 trace 采样帧 `s` 后（报告约定 `r=s+1`，只换算一次）：

| 量 | 窄台 15 例范围 |
|---|---:|
| 安全登顶 − 四足入台 | 16–22 |
| 安全登顶 − 0.4 代理首真 | 16–20 |
| 安全登顶 − 报告事件（换算后） | 16–20 |
| 失败事件 − 安全登顶（= 窗帧数） | 25–38 |
| 末有效物理帧 − 安全登顶 | 24–37 |
| 窗内低头比例 | 1.0（15/15） |

旧文“15–19”来自 1-based 报告步与 0-based trace 帧混算，Q037 §3 的质疑成立；本文 §4.3/§9 已按上表更正。

旧脚本的 `gate_step` 只是 `all(foot_on_platform)`（四足入台），**不是 0.4 m 门控起点**；0.4 门控首真由 trace 的 `top_arrived` 给出。有效物理样本：窄台每例 1074–1608 帧（每例剔除 1 条 post-reset 行，真实终止物理帧保留）；窗帧/低头比例只统计有效物理帧。

### 11.4 汇总错误修正

`report_top_cases` 原为 `len(rows)`=30，实际只有 **15** 例有非空旧 top（15 个宽台例为 null）；v2 的 `per_case.json.counts.report_top_cases=15`，并分别给出 `proxy_0p4_cases=15`、`four_feet_cases=15`、`safe_summit_cases=15`、`wide_real_safe_summit=0`。

### 11.5 派生件版本链与 SHA256

| 版本 | 产物 | SHA256 | 状态 |
|---|---|---|---|
| v0 | `logs/a2_v09/s2_0/reports/matrix-final-v7/report.json` | `c741a033ff03c481366c3d48f1f70fb31d4a0dab5bc21ee30f322036c77296d0` | 旧 0.4 代理，保留 |
| v1 脚本 | `logs/a2_v09/s2/q036_verify/recompute_s2_0_top.py` | `6ac93e1ce322a24be3d53353e620af5a74753a676169979a89c4773d4342329d` | 被 v2 取代 |
| v1 结果 | `logs/a2_v09/s2/q036_verify/s2_0_top_recompute.json` | `9060f963a098e537a4fb25eabc26c30644b0f0a6c0c85d60e87d7b9eca74aa1d` | 被 v2 取代 |
| **v2 脚本** | `logs/a2_v09/s2_0/verification/q037_top_recompute_v2/recompute_s2_0_top_v2.py` | `6946326e945f6462243fa53db90bb7d93f06fcd2082704431bf7d3d463b5532b` | 本件 |
| **v2 逐例** | `logs/a2_v09/s2_0/verification/q037_top_recompute_v2/per_case.json` | `5141459e83963070c6621e64bc7eef44cea7c00dafe5c1c786327a8dbe7d921d` | 本件 |
| v2 清单 | `logs/a2_v09/s2_0/verification/q037_top_recompute_v2/MANIFEST.json` | 含 30 例 trace/case、旧 v1、report/evidence SHA | 本件 |
| 派生报告 | `docs/v09_s2_0_top_recompute_q037.md` | 见该文件 | 本件 |
| 主 S2.0 报告 | `docs/v09_s2_0_diagnostic_final.md` | 已加 Q037 修订横幅与 §3 注 | 旧节保留标 superseded |

### 11.6 对签字的影响

- §4.3 的定性结论不变：窄台 15/15 在安全登顶后失败、窗内低头 1.0；宽台 0/15。
- 数字更正：窄台安全登顶窗为 **25–38 帧**（不是 43–55），安全登顶比 0.4 代理晚 **16–20 帧**（不是 15–19）。
- 该修订**不改变** Q036 主体判定：HEAD `cd18d44` 的 top 仍是 0.4 代理、WIP 原始信号修复仍需最终 SHA 的真实 GPU 验证；S2.0 的离线修订可由本派生件闭合（Q037 §3 已批准不重跑）。


## 12. 最终重审：`0a0276e`（Q036/Q037 全链复核与签字判定）

审计时间 2026-11-23 15:20–15:35 CST；主对象 `0a0276e0a2f6d1b3c5662f3d564d5736da4c85d6`（main，父 `cd18d44`）；Run A 在后续修复提交 `56820bba5ab9098889a439271aa060f2060746a5` 上重跑。工作区在审计期间另有未提交的 F-3/F-4 修复补丁（`metrics.py`/`s2_lifecycle.py`/`s2_quality.py`/tests），不属于上述 SHA。只读审计，未改生产/测试/SDS，未运行 GPU/训练。

### 12.1 被审文件与 SHA256

| 文件 | SHA256 |
|---|---|
| `src/tasks/velocity/v09/s2_quality.py` | `5fefffe94ef697b479f7614f6430d5cfd2c1e79e9a8b7bec84582c80a4495d30` |
| `src/tasks/velocity/v09/s2_lifecycle.py` | `d913b86815cd1ff08cde963ace53344f3d955e25dc0b89490af1d08c652b0ad0` |
| `src/tasks/velocity/v09/s2_train_real.py` | `a87eaed5fc6a3d66fe65ecedb1bb063a3b5d057c8ad2f2fc7e2d14110a8a922f` |
| `src/tasks/velocity/v09/s2_physics.py` | `242ca72960d70e8eef647253ce84f4e9aa12e572db13c60ac42f6e2552c0d917` |
| `src/tasks/velocity/v09/s2_acceptance.py` | `3c6b3d8574b6167a05212de6a83d0cdb0b345a04dafc317e4ca4a07f444af451` |
| `tests/test_v09_s2_quality.py` | `8b1ca610e29fa58395e20e3867bf67f27d3d95b2131b1d67845983e6c2f44205` |
| `tests/test_v09_s2_lifecycle.py` | `e11efd6a18df06bd71c43ec5e88b0fd7e53d246832f3ea282728f8d917ffd0a5` |

> 工作区状态：审计期间（15:15:47）出现未提交的 `s2_train.py`/`s2_train_real.py` trace 字段补丁（`s2_train_real.py` 工作区 `5ee70704…`，committed `a87eaed5…`），不属于 `0a0276e`；§12.4 的 Run A 崩溃来自该补丁。本报告对 `0a0276e` 的判定以 committed blob 为准。

### 12.2 逐项复核（检查单 1）

| 项 | 判定 | 证据 |
|---|---|---|
| T-1 零命令巡航窗不得成功 | **PASS** | `s2_quality.py:538-542`（`in_window &= cruise`）；`final_quality_crosscheck.json` 的 `T1_zero_command_cruise_must_fail`：`missing_window`、−10 |
| T-2 HOLD 锚定实际首零 | **PASS** | `s2_quality.py:586-600,740-758`；延迟 100 帧的用例 → `hold=900`、`hold_incomplete` |
| 站姿窗与首零 | **PASS** | instant 首零 401、终点 1400、hold=1000；ramp 首零 500、终点 1500、hold=1001（真实 trace，§12.4） |
| pitch 15°/10 帧窗作用域 | **PASS** | `s2_quality.py:556-559` 只在 `in_window` 计、出窗清零；边界：窗外 10 帧不锁、窗内 9 帧不锁、窗内 10 帧锁、恰好 15.0° 不锁、15.001° 锁、跨 8 s 窗不累计（6 例全过） |
| c_request / ramp 语义（Q037） | **PASS** | `s2_lifecycle.py:17,166-190`：请求帧 `c_request_after_step=0`、`c_request_at_actor=v`；ramp k=1..100，`k=100` 恰为 0；真实 trace `c_exec_consumed` 401=0.297…500=0 |
| HOLD 锚定（lifecycle） | **PASS** | `s2_lifecycle.py:187-204`；真实 instant/ramp trace 首零 401/500 |
| 单一奖励来源 | **PASS** | `s2_train_real.py:525-531`：reward 只读 `c_request_at_actor/c_exec_at_actor/phase`，`env.command` 仅由 lifecycle 写（`:351-353`），旧 `_s2_reward` stop 分支已删 |
| 门禁/终验共享判据 | **PARTIAL** | `s2_acceptance.py:18-25` 委托 `s2_quality.acceptance_reasons`（阈值/比较共享，`final_state_crosscheck.json` 边界 12/12 一致）；但 `metrics.py:160` 对楼梯仍用**全轨迹**窗计算 RMS/P99/over15，训练用 `entry+1 m→endpoint` 窗；同一“10 帧窗外 16°”训练通过而 gate 判 `pitch_over15`（`final_alignment_findings.json`）。Q036 A1 的“同一逐例判据”未完全成立 |
| 显式成功/失败终点 | **PASS** | `s2_lifecycle.py:200-243`；flat_walk 1000/instant 1400/ramp 1500；真实 flat_all：3×flat_walk +8、3×flat_stop −10、`timeout=false` |

### 12.3 独立参照对已提交代码的重放（检查单 2）

| 参照 | 结果 | 证据 |
|---|---|---|
| 24 个手算质量用例 | **24/24 PASS** | `final_quality_crosscheck.json`（直接 import 已提交 `s2_quality`，逐项比 outcome/endpoint/window/reasons/first_zero/cruise/stand） |
| T-1 / T-2 回归 | **2/2 PASS** | 同上 |
| pitch 边界 6 例 | **6/6 PASS** | 同上 |
| 时基 1000/1400/1500、首零 401/500、k100=0 | **5/5 PASS** | `final_timing_crosscheck.json`（生产 `S2Lifecycle`） |
| 同帧优先、pending/closed | **PASS** | `final_state_crosscheck.json`（物理失败/质量失败优先于成功；结束前 pending、终点 closed） |
| 逐 env reset / 跨 rollout 持久 | **PASS（CPU）** | 同一 `S2Quality` 2-env：env0 重置清空、env1 保留 401 窗样本/step=500，env1 在 step 1000 成功；`final_state_crosscheck.json` |
| 共享阈值边界 | **PASS** | gate/shared reasons 12/12 一致（含 5.0/5.001、9/10 帧、0.05/0.0501 等边界） |

### 12.4 真实 trace 审计（检查单 3）

| 证据 | 结论 | SHA256 |
|---|---|---|
| `stop_instant_final` | 请求 400、首零 401、hold=1000、终点 1400、**quality_failure、timeout=false**，reason `recovery+stand_peak_speed+stand_support`；`t0_rows=1`，`pre_step` 行不计 | trace `2e574746…` |
| `stop_ramp_final` | 请求 400、k=1@401=0.297、k=99@499=0.003、k=100@500=0（消费值）、首零 500、hold@1500=1001、终点 1500、quality_failure、timeout=false | trace `09ee7ea2…` |
| `flat_all_final` | 3×flat_walk 在 1000 成功；3×flat_stop 在 1400/1500/1400 质量失败（非 timeout） | trace `eadbbb31…` |
| `stairs_top_final` | 首个 0.4 m 门控帧 471（x=2.4228）时 `top_arrived=false`；真实安全登顶 **489**（x=2.8858，front=2.0，+0.8858>0.8）；524 物理失败；入台+1 m 窗未开启（max x=3.11），未覆盖登顶后 +8 | trace `6d71d9c4…` |
| 300 步 smoke | flat/stairs/platform 600/900/300 行，completed=true | 三者 SHA 见 `final_trace_audit.json` |
| **fresh 512 Run A（`56820bb`）** | **PASS**：2×24,576 行、46 字段；唯一 `(env_id,step)`、time-major 顺序 0 错；max\|ret−(adv+value)\| = 6.56e-7 / 4.77e-7；done 10/21、terminated=done、truncated=0、timeout=0；**reset 异常 0**（done 行 ep_len 为 reset 后的 0，下一行=1 且 g_level=0）；跨 rollout 连续 512/512、边界 reset 0；`c_request_at_actor`∈{0.3,0.5,0.6} 且 `c_exec_at_actor==c_request_after_step`；ramp_k=0、stop=false、first_zero=-1、hold=0（S2.1 96 步无停车）；quality 字段齐全，quality_ok 仅在 10/21 个物理失败帧为 false、其余 true；reward 有限（−10.2…0.05）；checkpoint SHA `d1aca5ba…`；隔离 ledger 副本 `ledger_shadow_final.jsonl`，正式 `ledger.jsonl` mtime 未变（13:49:25） | `final_run_a_audit.json` `9e270698…`；trace `39b57c7c…`/`e9d1d37e…`；log `8792e5b9…` |

**探针字段缺口（重要）**：保存的 4 份 q036 trace 是 15:01–15:03 由旧版 `run_stop_trace.py` 写出的 24 字段格式；committed 探针（15:04:43 之后）才写 `reward`、`reward_terms`、`c_request_at_actor`、`c_exec_at_actor`、`ramp_k_*`、`stop_request_step`、`four_feet_entry_step`、`cruise_samples`。因此实现文档 §4 引用的 at-actor/ramp/reward 数值**不能从已保存 trace 独立复核**，+2/+8/−10 的真实支付也未被 trace 记录（仅代码+CPU 单测）。

### 12.5 V13 CPU evidence 与测试套件（检查单 3/4 的旁证）

- `logs/a2_v09/s2/v13/v13_cpu_evidence.json` SHA256 **`3bf7aae71059d5c13d3b23c1c7c4a08463d8db80616e172207503ad9f5b1270e`**，`all_passed=true`，生成于 `2026-11-23T07:11:17Z`；但 checks 仍是原来的 7 项（handcalc/g_level/PPO/GAE/exception/r1/flat_anchor），**不覆盖 Q036 质量/top/时基**，`blocked_real_backend` 仍列 Run A。
- 我复跑实现文档的 8 个 S2 测试文件：**318 passed**；再加 `test_v09_s2_smoke_train.py`/`test_v09_s2_course.py`/`test_v09_s2_concurrency.py` 的 **18 passed**，合计 **336 passed**，与 lead 的“336 passed”一致。

### 12.6 剩余 FAIL（file:line）

| 编号 | 级别 | 问题 | 位置 | 证据 |
|---|---|---|---|---|
| F-1 | **已关闭（`56820bb`）** | 递归 `_py` 序列化 + 46 字段导出；fresh 512×2 块 Run A 通过（2×24,576 行，reset 异常 0，跨 rollout 连续 512/512） | `56820bb`：`s2_train.py` `_py/_arr`、`s2_train_real.py` 新增 info 字段 | `final_run_a_audit.json` `9e270698…`；trace `39b57c7c…`/`e9d1d37e…` |
| F-2 | **阻塞（证据）** | q036 探针 trace 缺 reward / at_actor / four_feet_entry 等字段；+8/−10 真实支付未记录 | `logs/a2_v09/s2/q036/run_stop_trace.py`（committed 版本新写，但保存 trace 早于它） | `final_trace_audit.json.probe_schema_gap`（12 个字段 cited_but_absent） |
| F-3 | **A1 不符** | gate 楼梯指标窗为全轨迹（`window=ones`），训练为 entry+1 m→endpoint；同一“10 帧窗外 16°”训练通过而 gate 判 `pitch_over15` | `metrics.py:160` vs `s2_quality.py:551-559` | `final_alignment_findings.json.gate_window_divergence`（gate `['pitch_over15']`，training `success`） |
| F-4 | **SDS 4.2 不符** | 训练成功距离用 `local_x - front`，SDS 4.2/Q019 A3 要求 `D_post = local_x - local_x(top_arrived)`；示范：front=2.0、真实 top=2.8858，lifecycle/quality 在 x=7.0 成功，规范为 7.8858（早 0.886 m） | `s2_lifecycle.py:141,208-214`；`s2_quality.py:629-641`；反例 `s2_cases.py:56`、`s2_metrics.py:475`；测试 `tests/test_v09_s2_lifecycle.py:29` | `final_alignment_findings.json.d_post_zero` |
| F-5 | 覆盖缺口（非错误） | 无自然通过的 stop、无自然安全登顶后 +8；通过分支仅 CPU 覆盖 | `docs/v09_s2_q036_implementation.md` §7 已承认 | `final_trace_audit.json`、`final_quality_crosscheck.json` |

### 12.7 自然轨迹 vs 受控边界

- **自然真实轨迹覆盖**：flat_walk 20 s +8 终点、flat_stop instant/ramp 的 400/401/500/1400/1500 时刻与质量失败 −10（非 timeout）、真实安全登顶事件（0.4 门控已开而 top 未开）、300 步三角色 smoke。上述均由 0a0276e 的真实环境代码产生（代码文件 mtime ≤14:58，trace 15:01–15:03；探针脚本字段更早，见 F-2）。
- **仅 CPU 受控边界覆盖**：通过的 stop 分支、安全登顶后 post-walk +8、逐 env reset/跨 rollout 持久、同帧优先/ pending-closed、pitch 边界、时基 1000/1400/1500。它们直接调用**生产 `S2Quality`/`S2Lifecycle`**（训练同一模块），但不是完整 `S2TrainEnv`+真实后端的受控驱动。
- **必需的最小受控真实路径 run**（若要求非 CPU 的通过分支证据）：在真实后端初始化“已停在 flat 的稳定站姿”与“已安全登顶、四足在平台”的状态（可用同一生产 `S2TrainEnv` + 受控初始状态/脚本化命令），跑到 1400/1400+ 或平台 +5 m 终点，记录 success/quality/reward；或由裁定方书面接受 CPU 覆盖。

### 12.8 签字判定

**CONDITIONAL（不签整链通过；签“实现级 Q036 A/top/B 通过 + Run A 通过”）。**

- **可签**：在 `0a0276e` 上，Q036 的两处已复现缺陷已修（T-1/T-2）、pitch 窗作用域与 15° 边界正确、c_request/ramp/HOLD/终点/单源奖励正确；独立参照 24+2+6 质量用例、5 时基用例、4 状态用例全部通过；真实 stop/flat/top trace 的时刻与“显式质量失败而非 timeout”成立；**Run A 在 `56820bb` 上通过**（2×24,576 行、46 字段、reset 异常 0、跨 rollout 512/512 连续）。
- **不可签**：整条 S2.1 训练链仍差 F-2（真实 reward/at_actor 字段未重采）、F-3（gate 楼梯窗仍为全轨迹 vs 训练 entry+1 m）、F-4（楼梯 +8 距离仍用 front，非 top_arrived）。三项已在工作区修复中（未提交），需在最终 SHA 上复核。
- **解除条件**：① 用 committed `run_stop_trace.py` 重跑 stop instant/ramp、flat_all、stairs_top，保留 reward/reward_terms/at_actor/four_feet_entry（F-2）；② 提交并在最终 SHA 复核 F-3 的共享窗与 F-4 的 top_arrived 零点；③ 若裁定接受现状则给书面例外。①② 完成后我重审并给最终签字。
- **依赖的证据 SHA256**：`v13_cpu_evidence.json` `3bf7aae7…`（只作旁证）；`final_quality_crosscheck.json` `156b2ebb…`；`final_timing_crosscheck.json` `4c960807…`；`final_state_crosscheck.json` `49a1f34f…`；`final_trace_audit.json` `153ae303…`；`final_alignment_findings.json` `7b3bfcc9…`；**`final_run_a_audit.json` `9e270698…`**；Run A trace `39b57c7c…`/`e9d1d37e…`、checkpoint `d1aca5ba…`、log `8792e5b9…`、隔离 ledger `11f4e699…`；5 份 q036 真实 trace 的 SHA 见 `final_trace_audit.json`；旧崩溃日志 `66d0e5f8…`（历史）。
