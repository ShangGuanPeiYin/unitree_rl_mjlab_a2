# Q009 v0.6：trial02 在 update 4224 按 SDS §5.6 子级预算停止（非崩溃）——停止分类与后续处置请示

- 时间：2026-11-17T12:35+08:00；提出：DSH 主对话。
- 合同：SDS `1ed4bf5fa8f83d4276e278c7d161be3b6d9c0032cbdc1e3540dade9606e6ad60` + Q001–Q008（Q008 `68caa982…`）。
- 源码闭包：`24c7781ed5c83af87bee4a40e7ad16e17632d0c2c0b4eb53e3413498c2b76709`（FREEZE v11 `03137aa4…`，未改变）。
- 活动 run：`logs/a2_ideal_stairs_v06/train_seed42_trial02_resume_000400_q007/`（已正常停止，进程退出，无异常栈）。
- 证据报告：`implementation/V06_TRAINING_STOP_4224.md`（`75da36c0…`）。

## 事实

1. **停止是规则触发，不是崩溃**。`stop.json`（`6bf0f7b2…`）：`stop_state=gate_failed`、`reason=substage_budget_exhausted`、`failure_class=training_failure`、`update=4224`、`substage=D25`、`spent=1500/1500`；`terminal.json`（`5e81ff19…`）`lifecycle_state=running`、`final_test=null`；`result.json`（`b208fe19…`）`budget_audit.consistent=true`。运行约 5h47m，无 traceback/OOM/exception.json。
2. **触发链（SDS §5.6）**：D25 在 4200 连续第 5 次门禁通过（`gates/validation_004200.json` `ccda2978…`，三速 20/20/20）→ `substage_advance` 到 D20；D20 新 generation 连续两个 64 例窗口 hard（`f=1.0, s=0.0`），第二个窗口在 4205 使 hard_streak=2（4204 末 `batches=1, cooldown=4`）→ `substage_regress` 回 D25；D25 子级预算已在 1500/1500（`transitions/004224.json` `6b1f7c6b…`：`spent_by_substage D25=1500, D20=24`）→ 按“返回已耗尽预算的子级，立即停止并报告预算失败”停止（4224）。
3. **D20 尝试被反复截断**：2700 进入 D；2900/3200/3500/3900/4200 五次 D25→D20 晋级，均在 4–5 次更新内（连续两次 hard 窗口）退回 D25；D20 五次合计仅 24 次更新。D20 是踏深 0.25→0.20 m 的加难（SDS 地形表 row16/17），窗口 `F=1.0, S=0.0` 说明策略未获得 0.20 m 踏深能力，退级本身符合“连续两次 hard → 退一级”的规定。
4. **D25 表现**：2700–4200 共 16 次晋级门禁，除 3700 当次 `c=0.3` 为 0/20、4100 `c=0.3` 19/20 外均 20/20/20；3700 失败后 gate_streak 归零并在 3800–4200 重新积累 5 次。保持集 flat 仅 800 一次 failed 2/15，其余全 0/15；H00 多次 failed 5–6/15（2400 达 10/15），均未达 12，streak 0。诊断 collapse 3850–4200 全 false。
5. **预算**：活动分支使用 4,224/30,000（剩 25,776）；正式累计 50（trial01）+50（Q005 丢失）+4,224 = 4,324/30,100。控制采样 103,809,024、物理子步 415,236,096、Adam 4,224。L/V/R 未进入；最终 300 例测试、视频、CPU 复算均未执行；`final_test/` 为空。
6. Q008 sidecar 已补至 4200（400–4200 共 38 条，`retention_interpretation_q008.jsonl` `9f072e75…`），未覆盖任何原始记录。Q008 生产修复按裁决仍未实施。
7. 过程事实：本段 14:08（上一轮）至 12:09 内没有接收/执行任何停止指令；`control.json` 仍为 `action=continue`（updated 1794867753）；DSH 未重启、未换 seed、未选择 checkpoint、未改动配方。

## 请求裁决

DSH 已按规则停止并保全证据，**未重启训练、未新建 fresh、未回退、未改动任何配方/代码**。请裁决：

- **A（默认建议）**：确认该停止为 SDS §5.6 的合法训练失败结论；本 v0.6 正式 seed42 结束（不出最终交付，不重启、不换 seed、不挑选更早 checkpoint 重跑）。DSH 只完成证据归档、Q008 记录语义修复规格（不落地）与最终报告（pass/fail/blocked/not_run 如实列出）。
- **B**：若主 Codex 认为“子级预算停止”在合同中应记为 `budget_exhausted` 而非 `gate_failed`，请明确最终报告的合同标签（DSH 本轮不改代码、不改已写死的 `stop.json`，只在报告与 sidecar 中标明实现标签与合同标签的差异）。
- **C**：若主 Codex 认为需要排查“D20 是否被单窗口 hard 提前退回”或“D20 加难顺序”等设计问题，请授权 **只读审计**（不训练、不改码）；DSH 可复算 4201–4205 的 64 例窗口、hard_streak 计数与 `_new_generation` 清空顺序，提交证据后再议。
- **D**：其他处置（例如是否允许后续在 **新版本/新裁决** 下做 D20 专项，而不是本 run 续训）。在得到明确裁决前，DSH **不**启动任何新训练。

## 边界

- 不修改奖励/PPO/课程/门禁阈值/速度/seed；不修改运行中或已停止 run 的原始证据；不 commit/push；R2 seed43 维持停止。
- 若裁决为 A，DSH 的后续工作仅限：证据完整性核对、`V06_TRAINING_PROGRESS.md` 收尾、Q008 修复的离线候选 diff/测试（按 Q008 §5 不落地）、以及最终报告。

## 附录（2026-11-17T12:45+08:00，等待裁决期间的离线进展）

- 源码完整性复核：`sha256sum -c V06_SOURCE_SHA256.txt` = **50/50 OK**；50 个冻结文件最新 mtime 为 2026-11-17 04:48（早于本段启动），未修改任何 `src/` 文件。
- 按 Q008 §5 交付离线候选（**未应用**）：`implementation/q008_candidate/`（diff `c435ee64…`，`patch -p1 --dry-run` 通过；候选测试 `3248e589…`；README `e3465c03…`）。在 /tmp 独立包中验证：保留集 `required=None`、每速 `passed=None`、`fail_required=12`、alarm/threshold 正确、`gate_eligible/gate_pass=False`；validation 18/20 与 final 95/100 不变；当前代码对同例会给出 `required=18`，候选测试在其上失败，证明测试有效。
- 停止后未做任何训练/GPU/代码改动；未 commit；R2 seed43 保持停止。请优先给出 Q009 裁决（至少 A/B 标签与是否允许后续新版本 D20 专项）。

## 附录 2（2026-11-17T12:40+08:00，只读复核结论）

- 我复核了 D20→D25 退级是否“单窗口误退”：`transition_004205.pt` 的 curriculum 状态显示 4204 末 `ema.batches=1`、`cooldown=4`（新 generation 已消费 1 个 64 例窗口），4205 再消费 1 个窗口才使 `hard_streak` 达到 2；这与 SDS“连续两次 hard 检查→退一个子级”一致。**不是单窗口误退**，候选 C 的“单窗口”疑点可关闭（若 Codex 仍需逐窗口 dump 可另派只读审计）。
- `spent` 记账复核：4200 D25=1481/D20=19；4205 D25=1481/D20=24；4224 D25=1500/D20=24，与 `spent_by_phase.D=1524` 一致；transition 事件 `detail.spent` 指向目标子级累计花费，无记账矛盾。
- 因此停止链的关键问题只剩**分类标签**（`gate_failed` vs “预算失败”）与**后续授权**，其余事实已可定稿。
