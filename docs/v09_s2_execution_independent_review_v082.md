# A2 v0.9 S2 执行链路 V8.2 独立复核（Q030 俯仰符号/派生/窗口/方法学）

- 审核 agent：A2-S2-VERIFY（独立验收子 agent，未参与实现）
- 审核时间：2026-11-22（快照 15:30–15:45）
- 仓库/分支/HEAD：`/home/xxs/dog/unitree_rl_mjlab_a2` / `main` / `2606f71`（DSH-Q030 留档）
- 范围：pitch 符号独立复算、T10 两分支窗口独立重算、派生字段/窗口口径/方法学标注；纯 CPU/只读证据。
- 机器可读件：`logs/a2_v09/s2_0/verification/execution_audit_20261122_r82.json`

## 0. 结论

| 项 | 结论 | 要点 |
|---|---|---|
| 1 pitch 符号约定 | **通过** | 三姿态独立复算与 `euler_heading` 一致；+X 前向、前髋 x=+0.25944 |
| 2 T10 real/plane 独立重算 | **通过** | real 44 帧/全低头/均值 −25.9595°；plane 18 帧/全抬头/均值 +14.1021°、范围 +12.2248～+15.9076°；与规划数字一致 |
| 3 派生字段 | **通过（旧产物仅留痕）** | `nose_down/up/level_fraction`、`pitch_sign_convention` 已实现；当前代码无 `positive_forward_lean_fraction`；旧 summary 保留并标注作废；211 测试通过 |
| 4 窗口口径 | **阻断** | 报告无独立四足入台 pitch 窗与 §6.5 正式巡航窗；未达窗口无 `NA` 标注；主 summary 的 `pitch_after_*` 仍含复位行 |
| 5 方法学/parity | **阻断** | 文档已写“病例匹配、物理状态未对齐”且不声称 Q016；但无同快照/同采样帧 Q024 输入；strict 1e-6 失败被写成 GPU 浮点噪声、1.75e-4 被 1e-3 容差掩盖 |
| **总判定** | **符号/窗口数值通过；窗口覆盖、Q024 同态、parity 归因阻断** | 数值可引用；口径与证据边界未闭合 |

## 1. 符号约定（独立复算）

- 独立公式：`(R_WB e_x)_z = 2(xz - wy) = -2(wy - zx)`；`pitch = -asin(2(wy-zx)) = asin((R_WB e_x)_z)`。
- 三姿态（绕 +Y 的 wxyz 四元数）：

| 姿态 | 四元数 | `(R e_x)_z` | −asin(2(wy−zx)) | v07 `euler_heading` |
|---|---|---:|---:|---:|
| 水平 | (1,0,0,0) | 0 | 0° | 0° |
| 抬头 20° | (cos(−10°),0,sin(−10°),0) | +0.34202 | **+20°** | +20° |
| 低头 20° | (cos(10°),0,sin(10°),0) | −0.34202 | **−20°** | −20° |

- `s2_env.PITCH_SIGN_CONVENTION` 写 `positive=nose_up, negative=nose_down, zero=level, formula='pitch = -asin(2(wy-zx)) = asin((R_WB e_x)_z)', forward_axis='+X', world_up='+Z'`，与上表一致。
- A2 XML：`FL_hip/FR_hip x=+0.25944`，`RL/RR x=-0.25944` → 前向 +X；与 Q030 一致。
- 结论：**符号与 +X 前向通过**；实现 `euler_heading` 的 pitch 就是 Q030 口径，未把绕 +Y 正旋转写成抬头。

## 2. T10 两分支独立重算（原始 trace）

### real-r4（`t10-stairs-real-r4/trace.jsonl`，1608 行）

- 选取 `step>=1563` 至最后未复位行（剔除最后 auto-reset 出生行），得 **44 帧，step 1563..1606**：
  - 全低头（pitch<0）：**true**；
  - 均值 **−25.9595°**，min **−53.1867°**，max **−11.7747°**；
  - nose_down/up/level = **1.0 / 0.0 / 0.0**；
  - 最后 step 1606、route **9.265788**、pitch **−53.1867°**。
- 事件：body failure **step 1608**；失败采样为非复位行 **1606**（文件行 1607），随后 reset 行 1607（文件行 1608）；`enter_climb 174`、四足 1560、top 1563。
- 与规划方 **44 帧、−25.96°** 完全一致。

### plane-r1（`t10-stairs-plane-r1/trace.jsonl`，1569 行）

- 首个机身 `route_x>=8.0` 为 step **1550**；至最后未复位行 step 1567，得 **18 帧**：
  - 全抬头（pitch>0）：**true**；
  - 均值 **+14.1021°**，范围 **+12.2248～+15.9076°**；
  - nose_down/up/level = **0.0 / 1.0 / 0.0**；
  - `top_arrived=false`；failure **step 1569**（nonfoot）；失败采样行 1568（step 1567，pitch +15.9076°），最后行 1569 为 reset。
- s−t=1、first replacement 1529、valid_fraction 1.0、saturation 0.025 与实现一致。
- 与规划方 **18 帧、+14.10°、+12.23～+15.91°** 完全一致。
- 差异说明：旧“+13.36° / 94.7%”是含复位行的 `pitch_after_platform`，且把 nose-up 误标为 forward lean；Q030 新数字为 **+14.1021°**，旧值作废。

## 3. 派生字段/旧产物/测试

- 当前 `scripts/run_a2_v09_s2_0.py::_pitch_stats`：`pitch_sign_convention`、`nose_down_fraction`、`nose_up_fraction`、`level_fraction`；分母为全部有效帧（零值计入），无 `positive_forward_lean_fraction`。
- `s2_env.PITCH_SIGN_CONVENTION` 存在；`scripts/recompute_a2_v09_s2_q030.py` 从原始 trace 重算并输出 `q030_pitch_recompute.json`（44/18 帧、step/row 配对、nose fractions、旧产物说明）；`post_reset.json` 标记旧 trace 的 reset 行（real 1607、plane 1568）。
- 旧 `smoke_summary.json` 仍保留 `positive_forward_lean_fraction`（real 0.0645、plane 0.947）与 `pitch_after_platform`（62/19 帧）；Q030 文档明确“旧 +13.36°/94.7% 作废、旧产物保留不删”，`q030_pitch_recompute.json` 也注明 superseded。**留痕通过**。
- 测试：`tests/test_v09_s2_0_q030.py` 2 项通过；S2 全套 **211 passed**；测试断言 nose fractions/零分母/符号。
- 结论：派生字段语义/命名/测试 **通过**；旧产物仅作历史保留。

## 4. 窗口口径（阻断）

- 报告 `docs/v09_s2_t10_case_selection_debug_ledger_parity_stairs.md`：
  - real 有 `four_feet_platform step1560`、`top_arrived step1563`、`step1563..1606 44 帧` pitch 窗；
  - plane 有前缘 `route>=8.0 step1550`、`top_arrived=false`、18 帧 pitch 窗；
  - 但**没有独立“四足入台后”pitch 统计窗**（real 的四足 1560 与 top 1563 之间 3 帧未单列），也**没有 §6.5 正式巡航窗**（`6.5` 在文档中无匹配）；未达窗口没有 `NA` 标注（plane top 直接写 false）。
- 主 summary 的 `pitch_after_platform`/`pitch_after_top` 仍从 `crossed[0]`/`top step` 到文件末尾，包含 reset 行（real 62 帧、plane 19 帧，旧值）；Q030 正确窗口只在 `q030_pitch_recompute.json`，主 summary 未同步。若后续报告引用主 summary，会回到旧口径。
- 结论：**窗口覆盖与主 summary 重置剔除未完成**（B-V82-1）。

## 5. 方法学与 parity

- 文档 §4.3 已明确：“normal real/plane 是独立 GPU 进程…**病例匹配但物理状态未对齐**，不能声明 Q016 满足”；并以 action replay 的 route/qvel 漂移为证。**方法学文字通过**。
- 但 **Q024 同态输入**要求同一物理快照、同一采样帧；当前 real/plane 仍是独立进程，文档也承认只能声明动作/命令前缀一致，**没有同快照双输入证据**；因此 Q016/Q024 的“同一状态”结论仍不成立（B-V82-2 证据缺口）。
- parity r6：`practical_ok=true`（容差 **1e-3**）、`strict_parity_ok=false`（1e-6）；obs max proprio **1.746e-4**、history 同、heightmap **5.96e-8**、route **1.624e-6**；首次 strict 分歧 step3 proprio **2.62e-6**。r4 容差 1e-5、practical=false（max proprio 2.66e-4）。同一量级差异靠把容差从 1e-5 放宽到 1e-3 变成 pass，**没有裁定依据**。
- 文档把 strict 失败写成“GPU float 噪声”；1.75e-4 比 float32 噪声（~1e-7）大约三个数量级，且 step3 已 2.6e-6，**更像采样相位/历史/归一化差异，不能归因于浮点噪声，也不应被 1e-3 容差掩盖**（B-V82-3）。
- 与实现方/规划方差异：规划方给的 44/18 帧、均值与范围与我们完全一致；差异只在“旧派生字段/旧 summary 是否被当作有效证据”和 parity 容差，不是窗口数值。

## 6. 判定

- **通过**：pitch 符号与 +X 前向；real/plane 独立窗口数值；nose fractions/pitch_sign_convention；旧产物留痕；测试 211 通过；文档“病例匹配、物理状态未对齐”文字。
- **阻断**：
  - B-V82-1：窗口未分列四足入台与 §6.5 巡航窗、无 `NA`；主 summary 仍含 reset 行。
  - B-V82-2：无同快照/同采样帧 Q024 输入，Q016 同态证据不成立。
  - B-V82-3：strict 1e-6 失败被写成 GPU 浮点噪声；1.75e-4 未定位且被 1e-3 practical 容差掩盖（r4 1e-5=false、r6 1e-3=true）。
- **待裁定**：四足入台/巡航窗的报告层级；parity 容差权威；Q024 同态方案（T11）。

机器可读明细见 `logs/a2_v09/s2_0/verification/execution_audit_20261122_r82.json`。
