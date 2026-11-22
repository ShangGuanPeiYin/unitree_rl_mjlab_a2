# V8.3 独立复核：四窗、pair、parity、matrix 与 Q031 反事实验收设计

- 复核角色：A2-S2-VERIFY（独立于实现方）
- 复核范围：`front_edge / four_feet / top_arrived / s6_5_cruise` 四窗、pair 证据、t11 维度 parity、matrix 批跑/恢复/失败/commit-release、首批 flat 用例、实时预算，以及 Q031 同帧双输入反事实的验收设计
- 复核方式：只读已有 CPU 证据、JSON、trace 和脚本；**未运行 GPU、未运行物理/训练、未修改实现、测试、SDS、ledger、checkpoint，未做 git 写入**
- 输出：本报告 + `logs/a2_v09/s2_0/verification/execution_audit_20261122_r83.json`
- 说明：matrix 仍在运行（快照 8/156、8,000 步），本报告只做阶段性复核；若后续出现超限、重复计费、失败未 release，立即单独报告

## 0. 结论摘要

| 项目 | 结论 |
|---|---|
| 四窗数值 | front 61/−20.065542、four 47/−24.992196、top 44/−25.959547、cruise NA；plane front 18/+14.102130、其余 NA；与实现方/父会话数值一致 |
| 四窗边界 | **待裁定**：four/top 的 `row_start` 比 raw 首个 flag 行晚一行；若 Q030 指“首个满足帧”，应为 48/45 帧 |
| `post_reset` / terminal | 均排除 reset 出生行、保留 terminal 样本；pitch 符号约定字段存在 |
| pair | **阻断（证据）**：`prefix_exact=false`，首次状态分歧 step1；**无 snapshot/restore 字段或证据**；full 的 plane 分支仅 350 帧，未到触发 step 1528；库中自述与证据不符 |
| t11 维度 parity | **支持状态先分歧**（step1 qvel[13]=4.77e-7），但 **不能解释旧的 1.75e-4**；r6 的 `practical=true @1e-3` 不是 strict parity 通过；文档仍以“GPU float 噪声”解释 strict 失败，未闭合 |
| matrix 运行器 | 恢复校验（status/case_sha/evidence SHA）和失败继续已实现；完整提交与进度一致；未见失败 |
| flat 完成用例 | 1000 步，trace/npz/case/SHA 绑定；四窗均 NA (`no_platform`)；`terminal_row_is_post_reset=false`；**通过** |
| 预算 | matrix 净 11,177 = full-r1 8,000 + limit2 3,177；另有 smoke 448 已 refund（384 reclass + 64 failed）；debug 8,066（含 reclass 384）；H 0；reserved 1,001；**未超限、未见重复计费**；但 pair 与 t11-dim **未计费**；旧 384 已由 reclass 解释 |
| Q031 | **尚未实现**；本报告给出同帧双输入反事实验收设计，验收状态仍为 pending |

## 1. 四窗独立复算

### 1.1 数值（`windows.json`）

| 分支 | 窗口 | 状态 | 帧数 | 起止 step | 均值 pitch/deg |
|---|---|---|---:|---|---:|
| real | front_edge | PASS | 61 | 1546..1606 | −20.06554238507063 |
| real | four_feet | PASS（边界待裁定） | 47 | 1560..1606 | −24.99219636289826 |
| real | top_arrived | PASS（边界待裁定） | 44 | 1563..1606 | −25.95954753754035 |
| real | s6_5_cruise | NA | — | — | 未到达巡航条件 |
| plane | front_edge | PASS | 18 | 1550..1567 | +14.102129741787495 |
| plane | four_feet/top/cruise | NA | — | — | no_platform |

- 与父会话给出的规划数值（real 44/−25.96，plane 18/+14.10）独立复算一致；front/four/top 的均值、起止 step 均与 `windows.json` 一致。
- `pitch_sign_convention` 字段存在，`pitch=-asin(2(wy−zx))=asin((R_WB e_x)_z)`，正=抬头；四窗统计均使用该约定。
- reset 出生行排除，terminal 样本保留；`terminal_row_is_post_reset` 与 trace 的 `post_reset` 标记一致。

### 1.2 原始行边界（新发现）

对 `t10-stairs-real-r4/trace.jsonl` 逐行复算：

| 条件 | 首个 raw flag 行 | 行内 step | 连续帧数 | 原始均值/deg |
|---|---:|---:|---:|---:|
| 四足均在平台 | 1559 | 1559 | 48 | −24.691044 |
| 到达顶部 | 1562 | 1562 | 45 | −25.631117 |

`windows.json` 使用事件 step 1560/1563 作为 `row_start`，因此比“首个满足条件的 raw 行”晚一行，帧数为 47/44。front 不受影响（61 帧一致）。

- 这是**边界口径问题**，不是数值计算错误：两套口径都能自洽，但不能同时称为“首个满足帧”。
- 若 Q030 定义为“进入窗口的首个物理帧”，应把 `row_start` 改为 1559/1562，并重算 48/45 帧；若定义为“事件确认后的下一帧”，则应在 `windows.json` 明写该约定。
- 在父会话裁定前，列为 **R-V83-1 待裁定**，不擅自改实现或证据。

## 2. Pair 证据

### 2.1 full（`t11-pair-stairs-full-r1`）

- real 1608 帧，plane 350 帧；`prefix_end_frame=350`；`prefix_exact=false`。
- `branch_prefix_sha256`：real `61b67a4bb8369e2d9c92d308af060102b34125d448947fe677ad4c7488c3aa30`，plane `8646c71250b8d9411c5f448ea2ae24d152b793d986e4a08125f57f57acd9ab0e`，前缀不同。
- 首次状态哈希分歧在 **step 1**；动作 SHA 因长度不同而不同。
- **没有任何 snapshot/restore 字段、restore 清单、恢复前后状态哈希或恢复证据**；`pair_summary.json` 的 `note` 与主文档中“snapshot/restore”自述缺乏证据支撑。
- plane 分支只有 350 帧，**从未到达触发步 1528**，因此 full 不能证明“同 case 在触发时刻的 plane 反事实”。

### 2.2 64 步 smoke（`t11-pair-stairs-64-r2`）

- real/plane 均 64 帧，动作 SHA 相等（`9543bb63…`）；`prefix_exact=false`，首次状态分歧 step 2；同样无 snapshot/restore。
- 动作相同而状态在第 2 步分歧，符合“动作一致、仿真状态先行分歧”的描述，但**不等于同帧双输入反事实**，也不构成 Q016/Q024 的通过。

### 2.3 结论

- `prefix_equal=false` 已按实记录；库中 `pair` 的 `prefix_exact` 字段与自述一致，但 **snapshot/restore 自述无证据**，full plane 未覆盖触发时刻。
- 列为 **B-V83-1 阻断（证据）**：在补齐 snapshot/restore 证据或删除该自述之前，pair 不能作为“同帧同状态”证据使用。
- pair 不满足 Q031 的“同帧双输入”要求；Q031 需要独立的反事实 runner，不能复用本 pair。

## 3. t11 维度 parity

`t11-parity-dim-r1/parity_summary.json`：

- 8 步；`parity_ok=false`，`strict=false`，容差 1e-6。
- 首次仿真状态分歧：step 1，`qvel` index 13，Δ=4.76837158203125e-07。
- 首次观测分歧：step 4，`proprio`，Δ=1.0132789611816406e-06；观测最大 proprio 差 4.649162292480469e-06。
- ledger：`obligation_id=s2_0_pipeline_debug`，`receipt=null`，`settled=null`。

判断：

1. 该结果**支持“状态先分歧、观测后分歧”**的定位，qvel[13] 是首个可指认通道；这是有价值的诊断证据。
2. 它**不能解释**旧的 64 步 1.75e-4：旧结果的通道、步号、是否同一 case/权重均未绑定，且本次 8 步最大 proprio 仅 4.65e-6，量级差约 37 倍。
3. 旧 `t10-parity-flat-r6` 的 `practical=true` 使用 1e-3 容差，`strict=false`，最大 proprio 1.746e-4；**1e-3 不能作为 parity 通过**，也不能用“GPU float 噪声”替代 strict 失败的归因。
4. 文档中“strict 失败是 GPU float 噪声”一句仍存在，和“case matched, physics not aligned”并存，属于**未闭合归因**；列为 **B-V83-2 阻断（待裁定）**：在父会话明确 1e-3 的定位（诊断阈值而非通过阈值）之前，不得把 r6 计为 parity 通过。

## 4. Matrix 运行器、恢复、失败与 flat 首批

### 4.1 运行器（`scripts/run_a2_v09_s2_matrix.py`）

- 156 个 entry（150 S2 + 6 H），每个 case 独立子进程、独立证据目录。
- `--resume` 通过 `valid_completed` 校验：`status=completed`、`case_sha256` 匹配、`evidence_files` 存在且 SHA 匹配；任一不满足即重跑。
- 失败不会中止整个 batch：失败 entry 记入 `failed_ids`，继续后续；结束返回非零并写 `matrix-summary-*.json`。
- ledger 仍由子进程按既有 reserve→commit/release 路径处理；matrix 层没有额外的 reserve，不会重复计费。
- **残余风险**：若子进程在 reserve 之后、release 之前崩溃，matrix 层只标记 failed，没有显式补偿 release 的路径；当前 8/156 未出现失败，列为 **R-V83-2 残余风险**，不构成当前阻断。

### 4.2 进度与提交

- 进度（最新快照）：completed 8/156，steps 8,000。
- full-r1 ledger commits：8,000；与进度一致。
- matrix 当前**净 spent**：11,177；其中 full-r1 8,000、limit2 3,177；gross commits 11,625，refund 448（384 reclass + 64 failed construction），未见 full 批次重复提交。
- 尚未出现失败，故“失败未 release”暂未发生；若出现，需按 B-V83-3 的预算口径立即核对。

### 4.3 flat 完成用例（`s2-000-real-aee59c5d3a49`）

- `control_steps=1000`；`settled={committed:1000, released:1, actual_samples:1000}`。
- `case_sha256=aee59c5d3a49a300c85a0a0067841059d36ea283ce1c0e62589477444e8bcef2`。
- `trace_sha256=fe1a7868d25d77159791a3f413b53c7ebf90fa0757c87229d3890461a5e1fe9c`。
- `npz_sha256=87c06637a7aacf3e7cb3333eb342f14139498d0dc1f6e77a70225da747f92c19`。
- 四窗均 `NA / no_platform`；`terminal_row_is_post_reset=false`；trace 逐帧 `post_reset` 字段存在且 1000 行全为 false（该用例无 reset 出生行）。
- 证据文件路径与 SHA 均在 `evidence_files` 中绑定；结论 **PASS**（flat 无平台，四窗 NA 合理，不是缺证据）。

## 5. 实时预算

| 项 | 值 |
|---|---:|
| matrix 净 spent | 11,177 |
| 其中 full-r1 commits | 8,000 |
| 其中 limit2 commits | 3,177 |
| smoke gross commits / refund | 448 / 448 |
| debug spent | 8,066 |
| H spent | 0 |
| reserved | 1,001 |
| envelope | 263,908,370 |
| 是否超限 | 否 |
| full 重复计费 | 未发现 |
| 未计费项 | pair full 1608+350、t11-dim 8+8 |
| 旧 384 去向 | 已解释：reclass 到 debug，并从 matrix refund |

- 当前 matrix 净 11,177 可由 8,000+3,177 完全解释；**没有超限**，也没有发现 full 批次的重复提交。
- V8.1 记录的 matrix 384 已由 ledger 解释：`reclass:matrix-smoke:refund` 从 matrix 退 384，随后在 `s2_0_pipeline_debug` reserve/commit 384；另有 64 failed-construction refund。故旧 384 不是丢失，而是重分类；**无重复计费**。
- pair 与 t11-dim 的 `receipt/settled` 为 null；若它们消耗了物理/训练预算，则当前预算是**低估**的。列为 **B-V83-3 阻断（预算）**：在补记账或明确为纯 CPU 诊断并记录依据前，不得宣称预算完整。
- H 预算尚未开始（0/6），matrix 全量仍在运行；本次不报超限。

## 6. Q031 同帧双输入反事实：验收设计

### 6.1 目标

在**同一帧、同一状态、同一观测、同一归一化器、同一 actor step** 下，仅切换 `c_exec` 输入，观察 `a_request/recorded_action` 的变化；同时设置独立干预控制，避免把“独立分支差异”误读为因果。该设计只用于诊断，不改变 Q031 的 `not_proven` 物理配对结论。

### 6.2 输入与当前缺口

现有 `--save-inputs` 已能保存：`proprio/history/heightmap/route_error/c_exec/step/refresh triplet/base_sample_z/plane_world_z/refreshed/front_foot_top_support/a_request/recorded_action`，以及 `case_sha256/weight_sha256/actor_step`。

仍缺：

1. `input_rows` 仍会 append terminal/reset 行，`post_reset` 只是元数据；反事实处理必须先显式过滤 reset 出生行。
2. 未保存 `c_request`（仅 `c_exec`）；无法区分“请求指令”和“执行指令”。
3. 未保存 normalizer / inference 身份（均值方差文件 SHA、推理入口、是否同一 policy 实例）。
4. 无同帧反事实 runner；现有 pair 是顺序双分支，不是同帧双输入。
5. 无动作灵敏度控制（重复同输入、零输入、阶跃输入、噪声底）。

### 6.3 验收步骤

1. **取帧**：从 covered frames 取帧；显式排除 reset 出生行与 post_reset 行；terminal 样本单独统计，不混入窗口。
2. **同帧重建**：对每帧冻结 `proprio/history/heightmap/route_error/base_sample_z/plane_world_z/refresh triplet`，确保两次推理使用同一状态对象与同一 normalizer。
3. **双输入**：
   - A：原始 `c_exec`；
   - B：干预 `c_exec`（零、阶跃或同幅反向）；
   - 同一 actor step、同一权重 SHA、同一 case SHA；分别记录 `c_request`、`c_exec`、`a_request`、`recorded_action`。
4. **独立干预控制**：
   - 对 A 重复推理两次，测确定性噪声底（应完全一致或给出固定噪声上界）；
   - 对 B 重复推理两次；
   - 用 A/B 的差与 A/A、B/B 的噪声底比较，只有显著超过噪声底才记为“输入可观测影响”；
   - 不以独立分支（real/plane）的差异代替同帧反事实。
5. **动作灵敏度指标**：n、RMS、max、zero fraction、动作逐维差、是否触发 clip；报告 sample 级与 case 级聚合。
6. **身份绑定**：case SHA、weight SHA、actor step、normalizer SHA、推理入口版本、refresh triplet、base/plane Z 全部写入每条记录。
7. **输出**：`q031_counterfactual.jsonl`（逐帧）+ `q031_summary.json`（覆盖数、排除数、噪声底、灵敏度、未覆盖原因）。

### 6.4 判定规则

- 仅对 covered frames 给出 `pending/verified`；未覆盖帧一律 `pending`，不得外推。
- 严格物理配对仍为 `not_proven`；本设计不改变该结论。
- **不得**用 1e-3 容差把失败改成通过；不得以“GPU float 噪声”作为归因；不得从独立分支差异推断因果。
- 若双输入动作完全相同且高于噪声底，只能说明该帧对 `c_exec` 不敏感，不能说明物理配对成立。
- 若 A/A 重复推理不一致，先修确定性/身份绑定，不得进入灵敏度判定。
- 若出现 reset 行混入、normalizer 不一致、actor step 不一致，整批作废，不进入评分。

### 6.5 当前状态

- Q031 实现尚未交付；本报告只提交验收设计。
- 现有输入捕获可作为基础，但 6.2 的 5 项缺口必须先补齐；否则同帧反事实不可审计。

## 7. 发现清单

| ID | 级别 | 内容 |
|---|---|---|
| B-V83-1 | 阻断（证据） | pair 自述 snapshot/restore 无证据；full plane 仅 350 帧、未到触发 step 1528；`prefix_exact=false`、首次分歧 step1 |
| B-V83-2 | 阻断（待裁定） | t11-dim 仅支持状态先分歧，不能解释旧 1.75e-4；r6 `practical=true @1e-3` 不是 strict parity 通过；文档仍以 GPU float 噪声解释 strict 失败 |
| B-V83-3 | 阻断（预算） | pair full 1608+350、t11-dim 8+8 未计费；若为物理/训练消耗则预算低估 |
| R-V83-1 | 待裁定 | four/top 窗口 `row_start` 比 raw 首个 flag 行晚一行（47/44 vs 48/45）；front 不受影响 |
| R-V83-2 | 残余风险 | matrix 子进程 reserve 后崩溃无显式 release 补偿；当前无失败 |
| R-V83-3 | 已解释 | V8.1 matrix 384 由 reclass refund 到 debug 解释；另有 64 failed-construction refund；当前净 11,177=8,000+3,177 |
| P-V83-1 | 通过 | 四窗数值、pitch 符号、post_reset 排除、terminal 保留 |
| P-V83-2 | 通过 | matrix resume 校验、失败继续、full commits=progress |
| P-V83-3 | 通过 | flat 完成用例证据 SHA 绑定、四窗 NA、terminal 标记 |

## 8. 证据清单

- `logs/a2_v09/s2_0/evidence/t10-stairs-real-r4/windows.json`
- `logs/a2_v09/s2_0/evidence/t10-stairs-real-r4/trace.jsonl`（SHA `4e43188e53e919683a2dc120b8bcc605ed79710ffb124fd9f7d5a0b25105ab4b`）
- `logs/a2_v09/s2_0/evidence/t10-stairs-plane-r1/windows.json`
- `logs/a2_v09/s2_0/evidence/t10-stairs-plane-r1/trace.jsonl`（SHA `e34c0c2e7f8a4dad70de4757487377286669eaefd2894daa37c5ab7d19841c5a`）
- `logs/a2_v09/s2_0/pair/0140473c4282/t11-pair-stairs-full-r1/pair_summary.json`
- `logs/a2_v09/s2_0/pair/0140473c4282/t11-pair-stairs-64-r2/pair_summary.json`
- `logs/a2_v09/s2_0/parity/t10/aee59c5d3a49/t11-parity-dim-r1/parity_summary.json`
- `logs/a2_v09/s2_0/parity/t10/aee59c5d3a49/t10-parity-flat-r6/parity_summary.json`
- `logs/a2_v09/s2_0/matrix/s2-000-real-aee59c5d3a49/r1/smoke_summary.json`
- `scripts/run_a2_v09_s2_matrix.py`、`scripts/run_a2_v09_s2_pair.py`、`scripts/run_a2_v09_s2_0.py`
- `src/tasks/velocity/v09/s2_physics.py`、`s2_state.py`、`s2_ledger.py`
- `logs/a2_v09/s2_0/verification/execution_audit_20261122_r83.json`

## 9. 第一阶段结论

- 四窗数值和 pitch 约定可复核；front 通过；four/top 存在**一行边界口径**待裁定，不能同时宣称“首个满足帧”与当前 47/44。
- pair 的 `prefix_exact=false` 已如实记录，但 **snapshot/restore 自述无证据**，full plane 未覆盖触发时刻；在补齐前 pair 只能作为“独立分支差异”证据，不能作为同帧证据。
- t11-dim 给出了首个状态分歧通道（qvel[13]）和量级，但不能解释旧 1.75e-4；旧 1e-3 practical pass 不得计为 parity 通过，strict 失败仍待归因。
- matrix 运行器、恢复、失败继续和 full commit 与进度一致；当前无超限、无 full 重复计费；flat 完成用例证据绑定完整。
- 预算：matrix 净 11,177 已解释，旧 384 已由 reclass refund 到 debug 解释；仍有 **pair/t11-dim 未计费**，在补记账前预算不能视为完整。
- Q031 尚未实现；本报告已给出同帧双输入反事实验收设计，等待实现方交付后复核。
