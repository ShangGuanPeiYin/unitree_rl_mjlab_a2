# s2.1-r2 关闭登记（DSH-Q038）

- 结论：`s2.1-r2` **工程不合规、未通过**；不得晋级、导出或作为正式恢复源（含 5M checkpoint）。
- 崩溃：rollout 12,965/31,726（40.87%），2026-11-24 14:25 CST，`torch.OutOfMemoryError`（PPO update epoch5 actor forward，20 MiB 分配失败；进程 5.73 GiB / 7.62 GiB，reserved-unused 856 MiB）。
- 退化证据（确定性复演，`scripts/record_a2_v09_s2_video.py`，S2TrainEnv 9 env、stairs N10、600 控制步）：入口 `s1_least` 正常登顶（x=+3.07 m，512 步 body 失败）；r2 5.01M 仍登顶（x=+3.19 m）；10.03M 起坍缩至起步区（x≈−0.64 m），20.05M/50.14M/100.27M/200.54M/315.85M 均未恢复。
- 账本实核：总 spent 338,927,616，其中 r2 318,627,840；reserved=0；余 461,072,384（=18,761×24,576+2,048），可用学习样本 461,070,336；无新增预算。
- 遗留：OOM 轮可能已部分 Adam 提交；后续须分别记录物理步、完整更新与持久 checkpoint 游标，废弃部分更新、不覆盖合法 checkpoint、不重复/退还计费。
- 授权范围：仅取证、合规修复、独立验收、既有额度内非学习验证；恢复训练须待规划裁定限额试训。
