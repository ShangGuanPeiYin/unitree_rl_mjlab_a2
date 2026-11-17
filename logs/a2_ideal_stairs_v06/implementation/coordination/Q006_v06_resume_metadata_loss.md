# Q006：trial02 恢复后 update150 因初始化元数据丢失停机，请裁决 101–149 的计费与继续方式

- 时间：2026-11-17T02:10+08:00；提出：DSH 主对话。
- 合同：SDS `1ed4bf5f…` + Q001 `5f3f2aa3…` + Q002 `3ebc7441…` + Q003 `45129da8…` + Q004 `768dc38a…` + Q005 `0159b032…`；最终闭包 `8eea96ae…5c90`；第 9 版验收 `accepted`/blocking=0（`d4b941a6…`）。

## 事实

1. Q005 桥恢复成功：从 `train_seed42_trial02/checkpoints/boundary_000100.pt`（`8ca0760e…`）恢复；`resume_mode=new_episode`、`replay_billed_updates=false`；update 100 validation 门禁 **passed（failed_cases=0）**、diagnostic 15 例运行；随后从 update 101 继续到 **149**（`commit_000101…149`，telemetry 51 条）。
2. 在 update 150 的**例行 checkpoint 保存**处停机：
   `RunnerError: the model has no ETH transfer metadata; v0.6 must start from the approved ETH Actor migration`
   （`runner._initialization_block` 读 `model.transfer_metadata`；resume 路径用裸 `V06ActorCritic()` + 权重加载，未恢复该元数据）。
3. 该异常路径的 `save('exception')` 同样因元数据缺失失败 → **恢复目录 `checkpoints/` 为空**，101–149 无任何可恢复模型状态（`exception.json` 记录 update=150）。
4. 修复（已实现，待验收）：`runner.load` 在加载模型后从 checkpoint 的 `initialization.transfer_metadata` 恢复 `model.transfer_metadata`（`deepcopy`），使恢复后的第一次保存可用；未改奖励/PPO/课程/门禁/seed。
5. 影响：trial02 现场目前只能从 `boundary_000100.pt` 继续；若如此，101–149 这 49 次更新会被**重新执行**（物理计算重复），而 run 内 `learning.update` 从 100 重新计。

## 请裁决

- **候选 A（建议）**：批准从 `boundary_000100.pt` 用修复后的闭包再次恢复（同一 trial02、同一 run）；把 101–149 的 49 次更新记为**工程损失**（`engineering_rework_updates=49`，控制样本 49×24,576=1,204,224、子步 4,818,816、Adam 980），单列不计入 trial02 的 30,000 学习预算，也不混入 30,050 的“两正式尝试”口径；物理总计算在台账中如实记录。trial02 仍为 100 + 29,900 的形式预算（in-run），不放宽单 run 30,000。
- 候选 B：不恢复 trial02，在 149 处终止；另批 trial03 fresh（需说明 combined cap 如何计：trial01 50 + trial02 149 + trial03 30,000 = 30,199）。
- 候选 C：其他处置。

## 边界

- 修复后的新闭包需要重跑 CPU/smoke/probe 与独立验收（同 SHA、blocking=0）后再恢复；不新建 fresh；不换 seed；不改配方。
- 在裁决前不启动新训练；trial02 目录、`boundary_000100.pt`、失败证据均保留；不 commit；R2 seed43 保持停止。
