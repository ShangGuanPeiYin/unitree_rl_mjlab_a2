# V8.4 独立复核：DSH-Q032 四窗起点、索引口径与边界

- 复核角色：A2-S2-VERIFY（独立于实现方）
- 复核依据：`doc/sds/reviews/v09_execution_rulings.md` 的 **DSH-Q032**（commit `7a54ee5`）
- 复核对象：`real-r4` 四窗、`windows.json`、主报告、`post_reset`/终止事件、旧 Q030 统计留痕；pair、parity、Q031 与 matrix/预算继续抽查
- 方式：**纯 CPU，只读现有 trace/JSON/脚本**；未运行 GPU、未推进物理、未训练、未修改实现/测试/SDS/ledger/checkpoint，未做 git 写入
- 快照时间：2026-11-22 16:15 CST；matrix 仍在运行（17/156）
- 产物：本报告 + `logs/a2_v09/s2_0/verification/execution_audit_20261122_r84.json`

## 0. 结论摘要

| 项目 | 结论 |
|---|---|
| real-r4 四窗数值 | **通过**：front 61、four 48/−24.6910°、top 45/−25.6311°（全低头）、s6.5 NA |
| 四窗起点边界 | **通过**：front/four/top 的起点前一帧均不满足，起点帧满足 |
| 索引口径 | **通过**：`trace_step`/`row_index`/`latched_event_step`/`event_completed_steps` 并列；1559/1560、1562/1563 是同一控制更新，不是 20 ms 锁存延迟；无全局 `step-1` |
| NA/失败/post_reset | **通过**：s6.5 与 plane 缺窗记 NA+原因，未记 0°；reset 行 1607 剔除；终止样本 1606 与 failure 事件保留 |
| 旧 44 帧留痕 | **部分通过**：旧 44 原件在 `q030_pitch_recompute.json` 保留；报告文字标 superseded，但该 artifact 内没有显式 superseded 标记，且报告仍指向它的旧 44 行 `step_row_pairs`，没有新的 45 行配对清单 |
| pair 证据 | **阻断（证据）**：报告称 pair summary 明写 `snapshot_taken=false...`，实际两份 summary 没有这些字段（读取为 null）；full plane 仍 350 步、未到触发 1528 |
| parity 逐维 | **状态先分歧已确认**：dim8 step1 qvel[13]=4.77e-7、obs step4=1.01e-6；dim64 obs step7=4.29e-6；dim4 为 obs-only 通过但 state hash 已不同；旧 r6 1e-3 已在报告标作废 |
| Q031 | **pending**：代码已补 `c_request`、normalizer 身份、reset/terminal 过滤和同帧双推理；四个代表组仍无 `inputs.npz`，无 GPU 采集，不能给 verified |
| matrix/预算 | **未超限**：17/156、17,000 步、0 失败；matrix 净 20,177 = 17,000 full + 3,177 limit2；debug 10,040（含 pair+dim 1,974）；H 0；无 full 重复计费；但 dim4/dim64 136 步未纳入 1,974，需核对 |

## 1. real-r4 四窗独立复算

数据源：`logs/a2_v09/s2_0/evidence/t10-stairs-real-r4/trace.jsonl`；`post_reset.json` 标记 row 1607=true，剔除；最后非 reset 物理样本 row 1606 保留；终止/failure 事件保留。

| 窗口 | 帧数 | trace step | 均值/deg | 最小/deg | 最大/deg | nose_down | nose_up |
|---|---:|---|---:|---:|---:|---:|---:|
| front_edge | 61 | 1546..1606 | −20.065542 | −53.186683 | +6.917234 | 0.934426 | 0.065574 |
| four_feet | 48 | 1559..1606 | −24.691044 | −53.186683 | −10.507992 | 1.0 | 0.0 |
| top_arrived | 45 | 1562..1606 | −25.631117 | −53.186683 | −11.180174 | 1.0 | 0.0 |
| s6_5_cruise | NA | — | — | — | — | — | — |

- four 首帧 route 8.336845；1 m 目标 9.336845；全回合最大 route 9.265788，未达，故 `s6_5_cruise` 正确记 `NA / s6_5_threshold_not_reached`，没有用 0° 或空窗代替。
- top 45 帧全部 `pitch<0`，即全部低头（Q030 语义下为前倾）；“登顶后持续前倾”结论维持。
- plane 复核：front 18 帧、1550..1567、均值 +14.102130、范围 +12.224783..+15.907557；four/top/cruise 均 NA（`four_feet_not_reached`、`top_arrived_not_reached`、`four_feet_not_reached`）。

### 1.1 两侧邻帧抽核

| 窗口 | 起点前一帧 | 起点帧 | 起点后一帧 |
|---|---|---|---|
| front | row 1545，route 7.980107 < 8.0，不满足 | row 1546，route 8.001490 >= 8.0，满足 | row 1547，route 8.024473，满足 |
| four | row 1558，flags [T,T,F,T]，不满足 | row 1559，flags [T,T,T,T]，满足 | row 1560，flags [T,T,T,T]，满足 |
| top | row 1561，top_arrived=false | row 1562，top_arrived=true | row 1563，top_arrived=true |

这与 `windows.json`（16:10:59 重新生成）的 `row_start` 1546/1559/1562 一致；不再以事件步 1560/1563 作为窗口起点。

## 2. 索引口径核查

代码事实：

- `src/tasks/velocity/v09/s2_physics.py`：控制更新后记录 `step=self.steps - 1`（gate 约 385 行、recorder 约 425 行）。
- `src/tasks/velocity/v09/s2_state.py`：`step()` 先 `self.completed_steps += 1`（约 328 行），随后 `_event()` 写 `step=self.completed_steps`（约 309 行）。
- 因此 trace row 1559/1562 是**该次控制更新对应的物理采样帧**，state event 1560/1563 是同一控制更新完成后的计数；二者不是 20 ms 的两帧，也不存在全局 `step-1` 修正。
- `windows.json` 现并列 `trace_step`、`row_index`、`latched_event_step`、`event_completed_steps`、`start_basis`；四足/顶部字段分别为 1559/1560、1562/1563，与上述解释一致。

待澄清（非阻断）：

1. `row_index=int(start)` 当前是“过滤后的 valid 列表下标”，不是原始 trace 文件行号。real-r4 只有末尾 reset，故两者相等；若未来 trace 中途出现 `post_reset`，`row_index` 会与原始行号分离，报告不应无条件称其为 trace row。
2. 每个窗口的 `threshold_frame_included` 对 front/four/top 为 false，对 s6.5 为 true；Q032 要求四窗“首真帧计入”，该字段的 false 容易被误读。实际帧范围正确，建议改为 `true` 或 `null`（不适用）。

## 3. 未达窗、post_reset 与失败

- `s6_5_cruise`：NA + `s6_5_threshold_not_reached`；未达原因是四足后净位移不足 1 m，不是缺数据；没有 0° 替代。
- plane：four/top/cruise 分别 NA + 原因；plane 在约 8.15 m 处 nonfoot failure，未 top。
- `post_reset`：real-r4 sidecar 仅 row 1607=true；windows 的 valid 范围 0..1606，1607 被剔除；plane row 1568=true 同样剔除。
- 终止：row 1606 的最后未复位物理样本保留并计入窗口；trace row 1607 的 failure/terminal 事件保留（event step 1608），报告仍记录 `failure: step 1608, reason=body`。未把终止行误当出生行。

## 4. 旧 44 帧/−25.96° 的留痕

- 旧统计仍可在 `logs/a2_v09/s2_0/evidence/t10-stairs-plane-r1/q030_pitch_recompute.json` 读取：`real_r4.selector='step>=1563 to last non-reset row'`、`frames=44`、`mean_deg=-25.9595`、`step_row_pairs` 从 1563 起。原件**保留**。
- 主报告文字已把旧 44/−25.96 标为 superseded，并采用 Q032 的 45/−25.6311；**“持续前倾”结论不变**。
- 但三个具体问题：
  1. 该 artifact 内没有 `superseded`/`replaced_by` 字段；`old_artifacts` 只说明旧 `positive_forward_lean_fraction` 的 smoke summaries 作废，没有单独标注 44 帧统计。
  2. 报告第 261/271 行仍写“`step/row` 配对逐条列在 `q030_pitch_recompute.json` 的 `step_row_pairs`”，但那里是旧 44 行（1563..1606），不是 Q032 的 45 行（1562..1606）。
  3. 旧 `windows.json` 已被原地覆盖；旧 44 数据只在 q030 artifact 留痕。若“原件”指原 windows 文件，则已丢失。

建议补一个 `q032_pitch_recompute.json`（或在该 artifact 加 `superseded`/`replaced_by` 与新 45 行 `step_row_pairs`），再让报告指向新产物；在此之前，第 4 项记为 **部分通过/证据待补**。

## 5. pair、parity、Q031 继续核

### 5.1 pair

实际 `pair_summary.json`（full 与 64）：

- full：real 1608、plane 350、`prefix_exact=false`、首次状态分歧 step1、plane 未到触发 1528。
- 64：64/64、动作 SHA 相等、`prefix_exact=false`、首次状态分歧 step2。
- 两份 summary **没有** `snapshot_taken`、`snapshot_restored`、`snapshot_status`、`restored_state_sha256`、`divergence_after_restore` 键；读取均为 null，且文件 mtime 仍为 15:53。

主报告第 347 行称“pair summary 明确写 `snapshot_taken=false, snapshot_restored=false, snapshot_status='not_implemented'...`”。代码 `scripts/run_a2_v09_s2_pair.py` 确实新增了这些字段，但**旧 summary 没有重新生成**，报告把代码能力写成了已有证据。列为 **B-V84-1 阻断（证据）**；full plane 未覆盖触发时刻的结论不变。

### 5.2 parity

最新文件：

- `t11-parity-dim-r1`（8 步）：state 首差 step1 qvel index13 4.768e-7；obs 首差 step4 proprio 1.013e-6；`parity_ok=false`、`strict=false` @1e-6。
- `t11-parity-dim4-r1`（4 步）：`parity_ok=true`、`strict=true`，但 `sim_state_prefix_equal=false`、state 首差 step1 qvel13 4.768e-7；这只是观测容差内通过，**不是物理状态一致**。
- `t11-parity-dim64-r1`（64 步）：state 首差 step2 qvel11 2.384e-7；obs 首差 step7 proprio 4.292e-6；false/false。
- 旧 r6：`practical=true @1e-3`、strict false、最大 proprio 1.746e-4；报告已明确“1e-3 practical pass 作废”。但 r6 artifact 本身的 `parity_ok=true` 仍存在，引用时必须同时引用其作废说明。
- 结论维持：**状态先分歧**，逐维定位到 qvel/其 history 副本；**严格物理配对 not_proven**；旧 1.75e-4 仍未被新证据解释。

### 5.3 Q031

- 代码侧已补：`c_request`、`actor_identity()`（normalizer state SHA、inference mode、action scale）、`input_rows` 过滤 `post_reset/post_step_terminal`、逐帧同帧双推理、real/plane 各重复推理的噪声底。
- `q031_input_coverage.json` 仍列四组（real-r4、plane-r1、limit-2 real、limit-2 plane）`inputs_present=false`，状态 `pending_gpu_capture`；无任何 GPU 采集产物。
- 因此 Q031 只能是 **pending**；不能写 verified，也不能用旧 trace 重建“同帧输入”；`physical_trajectory=not_proven` 维持。

## 6. matrix 与实时预算

### 6.1 matrix 进度

最新 `progress-matrix-full-r1.jsonl`：

- completed **17/156**，control steps **17,000**，failed **0**。
- resume 校验仍为 `status=completed` + `case_sha256` + evidence SHA；失败继续；当前无失败未 release。
- full commit 与进度一致；matrix 仍在运行，H 组尚未开始（0）。

### 6.2 ledger

| 项 | 值 |
|---|---:|
| matrix gross commits | 20,625 |
| matrix refunds | 448 |
| matrix net spent | **20,177** = 17,000 full + 3,177 limit2 |
| debug net spent | **10,040** |
| 其中 `q031:pending-physics:pair-dim` | 1,974 |
| H spent | 0 |
| active reserved | 1,001 |
| envelope | 263,908,370 |
| 是否超限 | 否 |
| full 重复计费 | 未发现 |

- `q031:pending-physics:pair-dim` 的 1,974 恰好等于 pair full（1608+350）加 t11-dim（8+8），说明上轮指出的 pair/dim 未计费已补记；但 pair/parity summary 中的 ledger 字段仍为 null，账目与 artifact 未互相引用。
- **新缺口候选**：`t11-parity-dim4-r1`（4+4=8）和 `t11-parity-dim64-r1`（64+64=128）共 136 步，未被 1,974 覆盖，summary ledger 也为 null。若它们是物理/GPU 运行，预算仍少记 136；若为纯离线分析，应给出 `skip_ledger`/纯 CPU 依据。列为 **B-V84-3 待核对（预算）**。
- matrix net 20,177 与 17,000+3,177 完全解释；无超限、无重复计费；refund 448 由历史 smoke/reclass 解释。

## 7. 发现清单

| ID | 级别 | 内容 |
|---|---|---|
| P-V84-1 | 通过 | Q032 四窗数值、起点、邻帧边界 |
| P-V84-2 | 通过 | trace/row/event 索引解释与代码一致，无全局 step-1 |
| P-V84-3 | 通过 | 未达 NA、post_reset 剔除、终止/失败保留 |
| P-V84-4 | 部分通过 | 旧 44 原件保留、报告文字标 superseded；“持续前倾”维持；artifact 无显式标记且报告仍指向旧 44 行 |
| B-V84-1 | 阻断（证据） | pair summary 实际没有 snapshot 字段，报告却称已明写 false/not_implemented；full plane 350 未到触发 |
| B-V84-2 | 阻断（证据） | Q032 的 `step_row_pairs` 报告指向旧 44 行 artifact，无新 45 行清单 |
| B-V84-3 | 待核对（预算） | dim4/dim64 共 136 步未纳入 1,974 commit，summary receipt null |
| R-V84-1 | 待澄清 | `row_index` 是 valid 列表下标，不一定是原始 trace 行号 |
| R-V84-2 | 待澄清 | `threshold_frame_included=false` 与“首真帧计入”语义冲突（数据范围正确） |
| P-V84-5 | 通过 | matrix 17/156、0 失败、无超限/重复计费 |
| P-V84-6 | 通过（pending） | Q031 代码字段补齐但无 GPU 输入，仍 pending/not_proven |

## 8. 证据清单

- `doc/sds/reviews/v09_execution_rulings.md`（Q032，commit `7a54ee5`）
- `logs/a2_v09/s2_0/evidence/t10-stairs-real-r4/trace.jsonl`（1608 行；SHA `4e43188e53e919683a2dc120b8bcc605ed79710ffb124fd9f7d5a0b25105ab4b`）
- `logs/a2_v09/s2_0/evidence/t10-stairs-real-r4/post_reset.json`（row 1607=true）
- `logs/a2_v09/s2_0/evidence/t10-stairs-real-r4/windows.json`（16:10:59 重新生成）
- `logs/a2_v09/s2_0/evidence/t10-stairs-plane-r1/trace.jsonl`（SHA `e34c0c2e7f8a4dad70de4757487377286669eaefd2894daa37c5ab7d19841c5a`）
- `logs/a2_v09/s2_0/evidence/t10-stairs-plane-r1/windows.json`
- `logs/a2_v09/s2_0/evidence/t10-stairs-plane-r1/q030_pitch_recompute.json`（旧 44 行）
- `logs/a2_v09/s2_0/pair/0140473c4282/t11-pair-stairs-full-r1/pair_summary.json`
- `logs/a2_v09/s2_0/pair/0140473c4282/t11-pair-stairs-64-r2/pair_summary.json`
- `logs/a2_v09/s2_0/parity/t10/aee59c5d3a49/t11-parity-dim-r1/parity_summary.json`
- `logs/a2_v09/s2_0/parity/t10/aee59c5d3a49/t11-parity-dim4-r1/parity_summary.json`
- `logs/a2_v09/s2_0/parity/t10/aee59c5d3a49/t11-parity-dim64-r1/parity_summary.json`
- `logs/a2_v09/s2_0/parity/t10/aee59c5d3a49/t10-parity-flat-r6/parity_summary.json`
- `logs/a2_v09/s2_0/evidence/q031_input_coverage.json`
- `logs/a2_v09/s2_0/matrix/progress-matrix-full-r1.jsonl`
- `logs/a2_v09/s2/ledger.jsonl`
- `scripts/run_a2_v09_s2_0.py`、`src/tasks/velocity/v09/s2_physics.py`、`src/tasks/velocity/v09/s2_state.py`

## 9. Q032 收口判断

- 四窗数值和边界已按 Q032 修正：front 61、four 48、top 45、s6.5 NA；起点前一帧不满足、起点帧满足；top 45 帧全低头；“持续前倾”维持。
- 索引解释与代码一致，1559/1560、1562/1563 是同一控制更新，不是锁存延迟；报告已并列 trace/row/event 字段。
- 未达 NA、post_reset 剔除、终止样本与 failure 保留均通过。
- 仍需修的两项证据链：
  1. pair summary 没有报告所称的 snapshot 字段（B-V84-1）；
  2. Q032 的 45 行 step/row 配对仍指向旧 44 行 artifact，旧 44 也没有 artifact 级 superseded 标记（B-V84-2）。
- 预算：pair/dim 已补 1,974，但 dim4/dim64 136 步待核对（B-V84-3）。
- 整批 156 条继续；窗口派生只用现有 trace，未重跑物理。matrix 若出现超限、重复计费或失败未 release，立即单独报告。
