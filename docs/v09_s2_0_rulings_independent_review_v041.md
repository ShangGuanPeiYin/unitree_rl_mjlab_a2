# A2 v0.9 S2.0-V4.1 聚焦复核报告

- 审核 agent：A2-S2-VERIFY（独立验收子 agent，未参与实现）
- 审核时间：2026-11-22（快照 14:30–14:35）
- 仓库/分支/HEAD：`/home/xxs/dog/unitree_rl_mjlab_a2` / `main` / `2a753da`
- 范围：F-V04-1 与五项跟进；纯 CPU，未跑物理/训练；未改实现/测试/SDS/台账、未 git 写。
- 机器可读件：`logs/a2_v09/s2_0/verification/rulings_audit_20261122_r41.json`
- 复测脚本：`logs/a2_v09/s2_0/verification/test_s2_v41_recheck.py`

## 0. 逐项结论

| # | 项 | 结论 | 关键证据 |
|---|---|---|---|
| 1 | F-V04-1：exact_clip 组汇总分母 | **通过** | 混合真实+替换：分子 1080 / 分母 1080 / 分数 1.0（不再 2160/0.5） |
| 2 | config_identity 声明与强制一致 | **通过** | 声明 11＝强制 11，含 stream/physics/reward/termination；逐项缺失均 ValueError |
| 3 | missing_reason / 原因码入记录与组汇总 | **通过** | `event_not_reached`、`no_valid_refresh` 入记录并单列计数；真实支 exact_clip 缺测原因保留 |
| 4 | 登记行双 SHA、反查、幂等 | **通过（附 1 项观察）** | 内容 SHA + 文件 SHA 均与原件相符；旧 SHA 两路可反查；重复写入不再追加；但 register 有 1 对重复行 |
| 5 | history_index_v2 只读派生与独立写入拒绝 | **阻断** | 重建确定性、磁盘与派生一致；但手工改条目+重算自哈希可被写入（temp 根实测 TAMPERED 落盘） |
| 6 | 回归 | **通过** | 84 项 S2 CPU 测试通过；V4 B-V03-1 回归仍为 frames=2/2 |

代码快照：`s2_map_ablation.py 17e2614d…`、`s2_metrics.py 994bdf29…`、`s2_cases.py 1744e94f…`。

## 1. F-V04-1：exact_clip 分母（通过）

用真实支（饱和分母 1080，exact clip NA）与替换支（远平面 raw=1.067，饱和/裁剪分子各 1080）混合：

| 用例 | 分子 | 分母 | 分数 | 可用例数 |
|---|---:|---:|---:|---:|
| 真实+替换（混合） | 1080 | **1080**（仅替换支） | **1.0** | 1 |
| 只有真实支 | 0 | None | None | 0 |
| 只有替换支 | 1080 | 1080 | 1.0 | 1 |
| 两支都未暴露 | — | None | None | 0；`missing_reason_counts={event_not_reached:1, no_valid_refresh:1}` |

- 真实支记录 `exact_clip_fraction=None`，未被计入；分子不再被真实支分母稀释。**F-V04-1 已修复。**
- 代码改为 `clip_den=Σ(saturation_denominator for clip_rows)`、`clip_num=Σ(exact_clip_numerator for clip_rows)`，逻辑与证据一致。

## 2. config_identity 与强制字段（通过）

- `config_identity().q024.required_pairing_fields` 与 `PAIRING_REQUIRED_FIELDS` **逐项相同**（11 项）：case/stream/weight/code/**physics**/**reward**/**termination**/state/command/action/sample_frame；另有 `shared_identity_fields` 7 项。
- 逐一删除 11 个字段后调用 `validate_pairing`，**11/11 均抛 ValueError**。
- 说明：V4 指出的“声明 9 项、强制 11 项”不一致已消除。

## 3. 原因码与记录/汇总（通过）

| 场景 | 记录 `missing_reason` | 说明 |
|---|---|---|
| 真实支未到事件 | `event_not_reached` | 与"未等到刷新"分开 |
| 替换支已触发、终止前无有效刷新 | `no_valid_refresh` | |
| 替换支未到事件即终止 | `event_not_reached` | |

- 组汇总 `missing_reason_counts` 对两类原因分别计数（实测 1/1）。
- `exact_clip_missing_reason` 已进入 `MAP_EVIDENCE_FIELDS`：真实支保留 `real_map_stores_only_clipped_height`；替换支零有效格时为 `zero_valid_denominator`（第三个值，见 Q-V41-2）。
- `normalize_map_evidence` 仍强制 23 个主字段，`exact_clip_missing_reason`/`missing_reason` 为可选并保留。

## 4. Q026 登记（通过 + 观察）

- 当前 register 两行；最新行同时含 `new_receipt_content_sha256=5ec1c897…` 与 `new_receipt_file_sha256=6f80b6fe…`，均与 `freeze_receipt_S2_DIAG.json` 实际内容/字节相符。
- 旧 manifest SHA 与旧 file SHA 均可反查同一新回执；旧/新清单自身 manifest SHA 与文件 SHA 全部核验通过。
- 幂等：在 register 副本上以同一旧/新 SHA 对再次调用，返回 `already_registered=True`，行数不变（无第三行）。
- **观察 O-V41-4**：register 中同一 old→new 对出现 **2 行**——旧的 14:25 行（无回执双 SHA）与 14:30 升级行。升级行有效，投影只取 1 条；但严格按 Q026 第 4 条“同一旧/新 SHA 对不重复追加”，这是迁移期重复。代码注释称旧行为 “legacy pre-Q026 row” 不准确（该行 ruling 已是 DSH-Q026，时间 14:25）。建议记录迁移原因或增加 version/supersedes_row 标记，不删除行。

## 5. history_index_v2 派生与独立写入（阻断）

- 派生正确：`history_projection(.)` 与磁盘 `history_index_v2.json` 逐字节一致；`projection_sha256=68ac664f…`；`read_only_projection=true`；`register_sha256=426b3ffc…` 与当前 register 一致；连续重建两次字节相同（确定性）。
- **B-V41-1（阻断）**：`write_history_projection()` 只校验必填字段、`read_only_projection is True` 和 `projection_sha256` 自洽，**不重新从 register 派生/比对**。在 temp 根实测：把手写投影的 `entries[0].reason` 改为 `TAMPERED`、重算 `projection_sha256`，调用 `write_history_projection` 仍成功落盘。即“禁止独立编辑”未被强制。
- 修法：写入前重新计算 `history_projection(root)` 并要求与传入投影完全相等（或只允许 `rebuild_history_projection` 写入、拒绝外部构造的字典）；补“篡改投影被拒”回归测试。

## 6. 回归

- `pytest tests/test_v09_s2_0_{cases,metrics,map_ablation,budget}.py` → **84 passed**。
- V4 的 B-V03-1 反例仍通过：t=30 触发、31 消费后终止、32 回调 → frames=2/2、终止后不计数、饱和分母 0/NA。

## 7. 待裁定/观察

**阻断**

- B-V41-1：历史投影的独立写入未被拒绝（见 §5）。

**观察**

- O-V41-4：register 有 1 对重复行（迁移产生）；旧行注释“pre-Q026”不准确；建议加迁移说明/版本标记。
- O-V41-5：exact_clip 缺测原因出现第三个值 `zero_valid_denominator`（零有效格），与两个暴露原因码不同类。

**待裁定**

- Q-V41-1｜Q026 第 4 条：迁移期重复行如何处理（保留+标记 vs 视为已登记）；不得删除 append-only 行。
- Q-V41-2｜Q023：`zero_valid_denominator` 是否作为 exact_clip 的正式缺测原因码（建议接受并文档化）。

## 8. 未执行

- 未跑 GPU/物理/训练、未写正式预留、未修改实现/测试/SDS/台账/checkpoint、未 git 写。
- 复现：`/home/xxs/anaconda3/envs/a2/bin/python` 运行 `test_s2_v41_recheck.py`；单元测试见 §6。

机器可读明细见 `logs/a2_v09/s2_0/verification/rulings_audit_20261122_r41.json`。
