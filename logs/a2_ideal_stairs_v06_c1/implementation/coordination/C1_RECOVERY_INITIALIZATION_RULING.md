# C1恢复初始化接线补充裁决

日期：2026-11-17；主协调转达主决策明确裁决，执行owner记录。

原方案仅批准速度(.3,.6)、D25/D20各4500、总36000和必要恢复接线；本补充不改变任何奖励、PPO、课程规则、随机分布或安全终止。

独立早审暴露执行owner新增两处接线错误：训练env显式frontier_tile_id启用了评估式单tile override；runner.load未在首次episode前恢复stage_iteration/generation。这两处必须修正，旧候选未运行GPU或正式训练。

批准的最小接线顺序：完整恢复checkpoint RNG → 原curriculum.schedule及phase_update一次 → 原ppo.reset一次。仅对带C1 continuation父链的load启用；普通旧合同恢复路径保持不变。训练env不传评估式override，角色数及允许地形仍由原schedule导出；评估显式override保留。

原set_schedule会对变化槽位内部初始化，随后原ppo.reset全量初始化。因此变化槽位允许初始化两次，明确是C1恢复初始化的额外随机消耗，不能声称无影响或物理位精确。不得前移schedule、倒回RNG、补seed、增加新的显式reset或落地生产RNG隔离修复。

验收必须记录首次及新ready二次load的各生成器恢复前/后、schedule后、reset后摘要，实际reset槽位与次数，512槽位最终role→tile、命令、阶段计数和generation68；临时reset不计学习、课程窗口或真实完成episode。D25例：flat0/frontier17/near15或16/easy1..14，角色数102/164/123/123。最终reset可按原机制重抽near/easy，不要求与schedule初表逐槽相等。检查模型/Adam/lr/normalizer保持、最终观察与奖励/终止历史属于最终episode；发现临时状态残留则上报，不扩大修复。

旧CPU保留为interrupted：388 passed、0failed、exit2，未完成完整套件。修正后必须新冻结并重跑完整CPU及GPU工程验证。
