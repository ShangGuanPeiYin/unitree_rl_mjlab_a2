# Q004：v0.6 seed42 首次训练在 update 50 因实现缺陷终止，请裁决重开方式

- 时间：2026-11-16T19:15+08:00；提出：DSH 主对话。
- 合同：SDS `1ed4bf5f…6ad60` + Q001 `5f3f2aa3…a4b1` + Q002 `3ebc7441…ccd52` + Q003 `45129da8…fa16`；训练已按 Q001-5 的固定 fresh 路径 `logs/a2_ideal_stairs_v06/train_seed42_trial01` 启动，冻结闭包 `2a5267a1…ce7d`，独立验收 D 第 3 版 `accepted`、blocking=0（报告 SHA `b0893032…83f5`）。

## 事实（有证据）

1. 训练于 19:05 启动，正常完成 50 次计费更新（F0，512 env、24,576 控制样本/更新），资源 measured 守卫在第 1 边界通过（整卡/进程/host/磁盘均达标），`telemetry/telemetry.jsonl` 有记录。
2. update 50（首个诊断边界）崩溃：`evaluation.training_state_snapshot` 对 `optimizer.state_dict()['state']` 使用 `enumerate(dict)`，把 int 键当 dict 调 `.items()` → `AttributeError: 'int' object has no attribute 'items'`。
3. 异常路径 `runner.save('exception')` 二次失败：`recovery.validate_curriculum` 报 `curriculum spend and learning.update disagree`（learning.update=50、spend 尚未提交），导致**没有任何 checkpoint 落盘**（`checkpoints/` 不存在）。
4. 目录 `logs/a2_ideal_stairs_v06/train_seed42_trial01` 已存在（仅 contract/control/resources/telemetry/exception/error），不能 resume（无 checkpoint），也不能按 Q001-5 复用该目录。
5. 未改奖励/PPO/预算/门禁/速度/seed；两个 bug 纯属实现层（快照迭代 + 提交顺序），不涉及配方或协议。

## 请求裁决

- **候选 A（建议）**：修复两个 bug → 完整 CPU → 重做 smoke/probe 证据绑定与新冻结（闭包会变）→ 独立验收 D 对同一新闭包重出 blocking=0 → 新建 **`logs/a2_ideal_stairs_v06/train_seed42_trial02`** 启动正式的 seed42 连续训练（仍 30,000 上限、同一配方/门禁/seed，不抽签）。
- 候选 B：若你希望保留“trial01 唯一 fresh 路径”的严格性，请指定如何处置已存在目录（例如由你授权一次 `--output-dir` 例外），我不会自行改断言或清空目录。
- 候选 C：其他处置（如暂停 v0.6 训练、先补 review）。

## 影响与边界

- 已计费 50 更新作废（无 checkpoint，不可恢复）；不选旧模型替代终点；不换 seed、不加预算、不改门禁。
- 我未删除/覆盖 trial01 证据；修复后会把新闭包/证据/验收 SHA 一并回执。
- 在收到裁决前不启动任何新训练。
