# A2 v0.9 S2.0-V4 裁定、登记、反例与预留独立复核报告

- 审核 agent：A2-S2-VERIFY（独立验收子 agent，未参与实现）
- 审核时间：2026-11-22（快照 14:24–14:34）
- 仓库/分支/HEAD：`/home/xxs/dog/unitree_rl_mjlab_a2` / `main` / `2a753da9c49a62a0c38e3d8c4fc63d49ca16a51e`
- 规范：SDS v1.2（`09fa2d4c…`）；裁定：`doc/sds/reviews/v09_execution_rulings.md`（Q026 + 四项落实确认）
- 机器可读件：`logs/a2_v09/s2_0/verification/rulings_audit_20261122_r4.json`
- 反例测试：`logs/a2_v09/s2_0/verification/test_s2_ablation_counterexamples_v04.py`
- 范围：纯 CPU；未做 GPU/物理/训练、未改实现/测试/SDS/台账/checkpoint、未做 git 写。

## 0. 结论分级

| 项 | 结论 | 关键事实 |
|---|---|---|
| Q026 留档（含四项确认） | **通过** | 文件 `f9f6e38b…`＝commit `2a753da` 内容；本地＝origin/main；Q026 与四项齐全 |
| 旧件字节不变 | **通过（V3 SHA 口径有局限）** | 旧 `S2_0.json` `09e2fc46…`、旧 receipt `0e888087…` 与 V3 一致；旧 history_index `42ce377d…` 大小/mtime 未变 |
| **B-V03-1 复查** | **已修复（通过）** | t=30 触发、31 消费后终止、32 回调 → frames=2/2、终止后不计数、饱和分母 0/NA |
| Q026 替代登记 | **通过**（receipt SHA 口径待明确） | `supersede_register.jsonl` 1 行；旧 SHA 可反查；同 SHA 对幂等；字段齐全 |
| 真实支/替换支 | **通过**（缺测原因落盘待补） | 真实支自有窗口、`replacement_applied=False`、无事件/无刷新记缺测；分支配对校验通过 |
| Q023 接线 | **阻断 1 项（组汇总）+ 声明一致性待补** | 23 字段+校验+attach/summary 就绪；混合分支 exact_clip 分母被稀释（F-V04-1） |
| 去重后正式预留 | **通过（未写入）** | 合计 263,888,370 ≤ 4 亿；S2.0 429,822 ≤ 0.5 亿；无重复计费；`q021_reservation_written=False` |
| **总判定** | **S2.0 输入/登记/预算通过；Q023 组汇总 1 项阻断** | 见 F-V04-1 |

## 1. Q026 留档与旧件核对

- 文件实测 `SHA256 f9f6e38ba1d79d68878bdb3414f14dbf523809b0e75633756274f91a5c7c607d`，与 `git show 2a753da:doc/sds/reviews/v09_execution_rulings.md` 完全相同；工作区无改动。
- commit `2a753da9c49a62a0c38e3d8c4fc63d49ca16a51e`（`docs: record ruling DSH-Q026 and its four confirmations`，14:24:57）＝`git ls-remote origin refs/heads/main`。
- Q026 与“四项落实确认”均存在，内容与回执一致：帧计数先判消费再计数、三元组、真实支自身窗口、L40 A/B 各一次缓存。
- 旧件核对（以 V3 记录为基准）：

| 文件 | 当前 SHA256 | V3 基准 | 判定 |
|---|---|---|---|
| `cases/S2_0.json` | `09e2fc4636ba6f0cd1d310abcf615eec751c7f8b020bdcd281e858deda6c7079` | 同（V3 §2.2） | 未变 |
| `cases/freeze_receipt.json` | `0e88808761f7080e235c2ffca3ac08239d617731b063abf50d61a4b5c9f50214` | 同（V3 工作期 14:08 观测） | 未变 |
| `cases/history/history_index.json` | `42ce377dc501e97da637fec44ced862775becfed3b251bbf58e35d29d48f7f91` | V3 报告未记录全 SHA | 大小 1635 B、mtime 14:15、内容仍 `suite=S2_0`，未变 |
| `cases/history/S2_0_v1_20261122.json` | `09e2fc46…`（同 S2_0.json） | — | 未变 |

局限：V3 报告/JSON 只记录了旧 `S2_0` 的 manifest/file SHA；旧 receipt 的全 SHA 来自 V3 工作期 14:08 的观测，旧 history_index 的全 SHA 未在 V3 落盘。本次已补记当前值，供 V5 起逐字节对照。

## 2. B-V03-1 复查（Q026 第一项）

当前实现快照：`s2_map_ablation.py` `98716269…`。反例结果：

| 场景 | 期望 | 实测 | 判定 |
|---|---|---|---|
| t=30 触发（未刷新）、31 Actor 消费后首次终止、32 回调 | frames=2、32 不改计数/不算消费、未暴露、饱和分母 0/NA | frames=2、终止后 frames=2、`consumed=False`、`finished=True`、`exposed=False`、`sat_den=0`、`sat=None` | **通过** |
| 触发当步即终止（t=10，consumed+terminal） | 计 1 帧、未暴露、NA | frames=1、finished、sat_den=0、NA | 通过 |
| 触发后未消费即终止（terminal 且 consumed=False） | 不增帧、未暴露、NA | frames=1（仅触发帧）、finished、sat_den=0 | 通过 |
| 批内 padding（consumed=False） | 不计数 | frames=1（仅真实消费帧） | 通过 |
| reset 后新病例 | 新帧只归新病例 | `reset()` 后 frames 从 2 归 1、分子重算；新对象互不影响 | 通过 |
| 顺序“先判消费再计数” | consumed=False 早返回 | 代码顺序正确（`if not consumed` 在 `frames_consumed += 1` 之前） | 通过 |

结论：**B-V03-1 已修复**；Q026 第一项落实。

## 3. Q026 替代登记

- 登记文件：`logs/a2_v09/s2_0/cases/supersede_register.jsonl`（690 B，1 行），位于旧 index 之外，只追加。
- 字段：`old_suite=S2_0`、`old_manifest_sha256=5efda5ab…`、`old_file_sha256=09e2fc46…`、`new_suite=S2_DIAG`、`new_manifest_sha256=fab7372e…`、`new_file_sha256=b990052c…`、`new_receipt_path=logs/a2_v09/s2_0/cases/freeze_receipt_S2_DIAG.json`、`new_receipt_sha256=5ec1c897…`、`reason="Q022 更名、旧清单零物理"`、`ruling=DSH-Q026`、`written_at`。
- 反查：`lookup_supersede` 以旧 manifest SHA `5efda5ab…` 或旧 file SHA `09e2fc46…` 均能定位同一新回执。
- 幂等：在临时根副本上以同一旧/新 SHA 对连续调用两次，均返回 `already_registered=True`，文件仍 1 行；不重复追加。
- 口径问题：`new_receipt_sha256=5ec1c897…` 是新回执的**内容 SHA**（`receipt_sha256` 字段），而新回执**文件字节** SHA 为 `6f80b6fe…`。Q026 第 3 条写“新回执路径/SHA”，未区分二者——见 Q-V04-1。
- 另：`history_index_v2.json` 与 `supersede_register.jsonl` 并存，属两套状态；Q026 第 2 条要求“避免多套可编辑状态”——见 Q-V04-4。

## 4. 真实支/替换支（Q026 第三项）

- `S2RealBranchEvidence` 已实现：用真实支自己的事件/刷新界定窗口（实测 t=10 触发、t=12 首个刷新 → `window_start_frame=12`、`s−t=2`）。
- 区分清楚：`branch='real'`、`replacement_applied=False`、`first_replacement_actor_step=None`、`replacement_frames_consumed=0`，不会写成已替换。
- 未到事件：`event_step=None`、`window_start_frame=None`、`exposed=False`、`note='not_exposed_before_terminal'`（"未到事件"与"未等到刷新"共用同一 note，见 Q-V04-6）。
- 真实支无法给出未裁剪高度：`exact_clip_denominator=None`、`exact_clip_fraction=None`，并给出 `exact_clip_missing_reason='real_map_stores_only_clipped_height'`；但该原因字段不在 `MAP_EVIDENCE_FIELDS` 中，进入病例记录时被丢弃（O-V04-8）。
- 分支配对：`validate_branch_pair` 强制 7 项共享身份（case/stream/weight/code/physics/reward/termination）完全相同，仅允许 `diagnostic_map_branch`/`record_id` 差异；实测物理身份不同即报错。

## 5. Q023 接线与反例（Q026 第二项 + 阻断项）

- 病例记录接线：`s2_metrics` 提供 `MAP_EVIDENCE_FIELDS`（23 字段）、`normalize_map_evidence`（缺项/类型报错、NA 不填 0）、`attach_map_evidence`、`summarize_map_evidence`；含 `replacement_applied`、`replacement_frames_consumed`、`sample_reference_count`、`sample_references_sha256`、`first/last_sample_reference`。
- 三元组与消费：`native_sample_id={refresh_index, sample_sim_time_s, produced_at_actor_step}` 严格校验；`event_step`、`first_replacement_actor_step`、`s_minus_t` 均在 summary；每个消费帧可追加 sample reference。
- 最小证据“声明即校验”：`validate_pairing` 缺项报错（实测缺 10 字段即 ValueError），`validate_native_sample_id` 严格要求三元组，`normalize_map_evidence` 缺 23 字段中任一项即报错；`validate_branch_pair` 校验跨支身份。**但** `config_identity` 对外声明的 `required_pairing_fields` 仍是旧 9 字段元组，未包含 stream/physics/reward/termination（O-V04-7）。
- 物理/奖励/终止配置身份：配对字段要求并提供校验，但模块本身不产出这些 SHA（应由 runner 提供）；当前无 S2 runner，故"产出"仍待端到端（O-V04-6）。
- 单元测试：`tests/test_v09_s2_0_{cases,metrics,map_ablation,budget}.py` **79 passed**。
- **F-V04-1（阻断，组汇总）**：`summarize_map_evidence` 用"全部记录 saturation 分母之和"作 `exact_clip_denominator`，而 `exact_clip_numerator` 只来自有未裁剪数据的支。实测混入真实支记录时：真实 1080 + 替换 1080 → 分母 2160（应为替换支 1080），只要分子非零，`exact_clip_fraction` 就被稀释 2 倍。按分支分别汇总可规避，但函数未强制同一 branch。修法：强制同 branch，或按各行 `exact_clip_denominator` 非空求和；补混合分支回归。

## 6. 去重后的正式预留（Q026 第四项）

`q021_reservation_written=False`（未落账）。最终逐项表（`s2_budget.s2_stage_reservation_draft`，新 `S2_DIAG` 清单）：

| 项 | 控制步 | 去重核对 |
|---|---:|---|
| 1024 探测 | 3,709,184 | 与 512 独立；含 5 热身+50 测量+1 恢复+评估+录像 |
| 512 探测 | 2,332,928 | 同上；两档均预留，未重复计 |
| S2.0 两分支 | 420,390 | 150 条原生轨迹；H 另计 |
| H 六例入口补采 | 9,432 | 与 S2.0 矩阵不重叠 |
| H 候选复测（3 模型） | 28,296 | 3×9,432；与三套终验的 S1 对照不同 |
| L40 里程碑冻结对照缓存 | 438,440 | A/B 各 219,220；与门禁内候选里程碑运行不重叠 |
| ≤40 次门禁（全按最坏 S2.3） | 173,940,000 | 40×4,348,500；上界保守，不重复 |
| 三套终验+冻结对照 | 83,009,700 | 每套 27,669,900（候选 2700+S1 2700+6400 900） |
| **合计** | **263,888,370** | **≤4 亿；余 136,111,630** |
| S2.0 子额度 | 429,822 | ≤0.5 亿；余 49,570,178 |
| 学习 | 32 亿 / 36 亿 | 余 4 亿=两次 2 亿恢复 |

- 等待空步/padding/终后 reset 排除/S1 不复用：`billing` 字段均声明；逐项无重复计费。
- 正式预留仍未写入；按 Q021 第 1 条，S2 物理启动前必须由主对话下令写入。

## 7. 待裁定与观察

**阻断**

- **F-V04-1**（Q023 第 5 条）：`summarize_map_evidence` 的 `exact_clip_denominator` 混入真实支分母。证据：真实 1080 + 替换 1080 → 2160；应仅取替换支 1080。按分支汇总可规避，但函数未强制。

**观察**

- O-V04-1 旧 `S2_0`/旧 receipt 与 V3 一致；旧 history_index 大小/mtime 未变（V3 未记录全 SHA，本次补记 `42ce377d…`）。
- O-V04-2 登记 1 行、字段齐全、旧 SHA 可反查、同 SHA 对幂等。
- O-V04-3 登记用的新回执 SHA 是内容 SHA，不是文件 SHA。
- O-V04-4 B-V03-1 已修复，Q026 第一项通过。
- O-V04-5 真实支独立窗口、不伪装已替换、分支配对校验通过。
- O-V04-6 Q023 字段/校验/attach/summary 就绪，79 项 S2 CPU 测试通过；无 S2 runner，端到端病例记录与配置 SHA 产出待物理阶段。
- O-V04-7 `config_identity` 声明的配对字段清单仍是旧 9 项，实际强制 11 项。
- O-V04-8 真实支 `exact_clip_missing_reason` 在 metrics 记录中被丢弃。
- O-V04-9 预留草案 263,888,370，未写入。
- O-V04-10 `history_index_v2.json` 与 `supersede_register.jsonl` 两套状态并存。

**待裁定**

- Q-V04-1｜Q026 第 3 条：登记的新回执 SHA 用内容 SHA 还是文件 SHA（或两者都记）。
- Q-V04-2｜Q023 第 5 条：混合分支组汇总的 `exact_clip` 分母口径（建议强制同分支并按行求和）。
- Q-V04-3｜Q023/真实支：`exact_clip_missing_reason` 是否纳入病例记录字段。
- Q-V04-4｜Q026 第 2 条：v2 index 与 register 并存是否可接受（建议声明 register 为唯一权威）。
- Q-V04-5｜Q024 第 5 条：`config_identity` 的 `required_pairing_fields` 更新为 11 项。
- Q-V04-6｜Q026 第 3 条：真实支"未到事件"与"未等到刷新"是否需不同原因码。

## 8. 未执行与复现

- 未做 GPU/物理推进、未训练、未补采、未写正式预留、未修改实现/测试/SDS/台账/checkpoint、未 git 写。
- 反例：`/home/xxs/anaconda3/envs/a2/bin/python` 运行 `test_s2_ablation_counterexamples_v04.py`。
- 单元测试：`pytest -q tests/test_v09_s2_0_{cases,metrics,map_ablation,budget}.py` → 79 passed。
- 登记幂等：临时根副本上 `register_supersede` 连续两次 → `already_registered=True`、仍 1 行。
- 冻结件校验：`validate_s2_0_manifest`、`validate_s2_diag_run_plan`、`validate_h_series_manifest` 均为 True（快照 14:34）。

机器可读明细见 `logs/a2_v09/s2_0/verification/rulings_audit_20261122_r4.json`。
