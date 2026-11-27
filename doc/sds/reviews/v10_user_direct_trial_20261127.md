# 用户直接启动限额试训的新指令（2026-11-27）

用户在实施主对话最新明确指令：“直接尝试启动训练”；随后补充“如果1024环境不行的话就尝试512环境”。这取代先执行资源v2与112诊断才启动的本次前置顺序；不伪称这些门禁已通过，完整验收仍未通过。

本次先用现有d542源码、原A1 teacher及原奖励/损失启动D1/N1024/48步rollout、最多50 retained更新，charged硬帽2457600，沿formal_A1_20261127_approved/learning_costs.jsonl和同canonical trial guard/id。保存后停止，不自动课程/PPO，teacher_admitted=false/accepted=null。已有成本不退款、不重置。512仅作为1024失败后的已授权备用；现guard固定N1024，若需切换必须显式技术迁移并沿同账本/原硬帽，不能换id赠额度或把50更新暗改100。

启动前本轮只读观测MemAvailable约16.5GiB，原resource/teacher失败证据保留；本次是用户直接授权的限额尝试，不是SDS完整准入或行为验收通过。
