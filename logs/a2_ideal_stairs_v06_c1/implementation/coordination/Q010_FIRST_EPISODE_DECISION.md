# Q010：一case一首episode评估证据修复

日期：2026-11-17。主协调转达主决策明确批准，执行owner记录。源C1方案SHA：22d3c77a6af62042eb98f8b310ea3d1d7e8554b77a811ad618be2a4e0a35c34c。

## 已证明事实与边界

固定v2的environment_case_runner将每次终止写入episodes[env_id]，批量等待较慢槽位时，同槽自动reset后的后续episode可覆盖首个结果；输出steps是批量循环步数，未透传首回合mean_vx。受控CPU反例（FIRST_EPISODE_OVERWRITE_COUNTEREXAMPLE.json）显示首fail后success会报pass，反向也可翻转；这证明实现语义问题，不证明历史真实GPU门禁全部受影响或无效。原env._episode_records已有完整终态字段。

## 批准的生产修复

- 仅v06生产environment_case_runner修复一case一首episode语义，不只在logs基线替代。首次终止pre-reset记录深拷贝；其后同槽episode不覆盖、不补字段，不与后续live diagnostics混合。
- 严格case/env一一对应和索引，missing/越界/重复env_id、缺必要首回合字段、非有限核心证据均报无效，禁止借后续episode凑齐。
- 每case.steps使用首回合真实控制步数，诊断active/stall/first_step同属该首回合；mean_vx/mean_speed仅报告，不加实际速度门槛。
- 批量中仍存活槽位的动作、终止与物理推进保持原样。等待慢槽时额外自动reset及控制计算单列：实际控制=B×循环步数，有效case控制=各首回合steps之和，额外=二者差，物理子步各乘4；不计正式PPO/课程预算。
- 普通门禁18/20、最终95/100、保持15例/12失败/连续两次、所有速度/出生范围/奖励/PPO/安全规则不改；Q008记录语义与完整RNG隔离不在本项修复。

## F0第500步补充批准

F0训练物理上限仍1500步。未曾首次终止且存活到逻辑500步的槽，深冻同帧info(success/failure/timeout/reason)和其当前diagnostics为window_end，steps=500；500同帧若有物理终止，首terminal优先；早失败永远使用首终态，不混入后续episode的500窗。

仅批准diagnostics_snapshot（diagnostics_cases逐槽出口）新增两个只读字段mean_vx=sum_vx/max(episode_length_buf,1)、mean_speed=sum_speed/max(episode_length_buf,1)，与_episode_records同义；不新增累计量、坐标变化或随机操作。缺同帧记录/非有限仍无效。净位移门禁>=5c-1e-9与Q005一致性atol1e-6/rtol0保持。

## 验收与成本

先RED首失败后成功/反向、8槽异步、深拷贝、索引/缺字段/NaN、499失败/500失败/500存活、诊断分母和真实env累计均值。修复后新冻结、完整CPU、同SHA新smoke2/probe56、30例基线必须走修复后的生产runner，logs观察器只读核对，不替代结果；独立blocking=0才开训。

v2完整CPU483 passed/0failed/exit0及已完成smoke2/probe56均保留为历史工程证据，不能充当Q010修后放行。v2实际工程成本58更新：1377024控制样本、5508096物理子步、1160Adam；修后授权再做一轮58工程更新，正式cap仍36000/累计36100，不新增正式损失或重放4224。

## 后续明确批准：新run证据持久化与失败去重

新run的每次gate/retention/diagnostic，以及最终test，保存同次实际result.records的深冻JSON sidecar；case_table_sha/records_sha/summary_sha复用evaluation规范，落盘复算须一致，主记录绑定路径/SHA。旧run不补造，不把全部历史塞checkpoint，不重评/重采样/重置训练RNG。sidecar原子不覆盖；既有匹配文件重试补齐文件和目录fsync，冲突拒绝。

单当前evaluation_transaction只为同一待持久化结果留证，身份绑定source/contract、update/phase/substage/generation、kind/table、model/optimizer摘要。evaluated表示实际评估完成、应用尚未开始；applied表示已应用且必须匹配gate/diagnostic/final及保持对应记录。只有完整checkpoint含应用状态及pending result身份后清除缓存。写失败→异常checkpoint→正式裸load应续写相同结果，不再调用case_runner、不重计PPO/streak。应用中途失败标application_incomplete，允许保存真实证据，正式恢复拒绝并上报，禁止猜测回滚或补计。若保存证据自身不能持久则停留问题，不宣称无损恢复。

## 后续明确批准：GPU保持协议零学习验收

最终冻结额外使用原4224独立均值模型，flat/H00各15例、三速(.3,.5,.6)各5、原standard表，复用生产environment_case_runner。完整记录first terminal或500步window_end、原reason/mean、有效case控制与实际batch控制及额外物理成本；不进入正式课程/streak/PPO预算。F0必须报告实际500步window_end与terminal例数；若全部提前终止，该GPU分支未建立，保留结果报主协调，不自行重试挑结果或改物理。

前文“diagnostics_snapshot（diagnostics_cases逐槽出口）”仅指diagnostics_cases方法，实际修改仅该方法新增mean_vx/mean_speed两字段，未扩大diagnostics_snapshot旁路。
