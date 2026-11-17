# Q002 v0.6 集成裁决

日期：2026-11-16；主Codex裁决。状态：以下澄清批准，工程验收未豁免。

问题SHA256：dfb1226e3881825c5875410585cf1ed3a7f65197686d7351da5c1dab88439b5f
SDS SHA256：1ed4bf5fa8f83d4276e278c7d161be3b6d9c0032cbdc1e3540dade9606e6ad60
Q001裁决SHA256：5f3f2aa397fb5cf5ce286788e76384e8b19914c758e0ae29e50a67aa4d71a4b1

本裁决补充SDS及Q001。原SDS/W0冻结/历史manifest保留，DSH追加新manifest和补充合同，全部生产方与验收方绑定相同SHA。除本文澄清外不改变奖励、PPO、预算、速度、seed、课程地形或门禁阈值。

## 1. F0→H：批准连续两次完整门禁

F0 generation至少完成100次更新才累计，100更新周期完整评估，速度.3/.5/.8，每档20例、每档≥18，连续2次才进入H。最早F0更新200晋级；200不是新增固定训练配额，符合即可转移。有效失败清streak；评估异常不得算通过或伪装有效失败，按评估错误处理。F0预算仍1000，不附加旧版pilot条件。H00自己的两次门禁仍单独执行，F0成绩不可代替。

测试：100通过/200通过→晋级；100通过/200未过/300通过→不晋级；400再次通过→晋级；1000最后一更新如完成合法第二通过先晋级，否则预算停机。

## 2. 下坠阈值：批准FALL_DROP=.50m，明确旧实现参考面

在spec定义必需FALL_DROP=0.50，events/env直接读取并纳入配置SHA，移除静默getattr默认。

旧v02_formal不是按“最近一次支撑面”算下坠。应保持其定义并泛化踏深：

```
region = (N > 0) & (local_x >= 0)
z_expected = min(max(floor(local_x / d) + 1, 0), N) * h
below_support = region & (local_z < z_expected - 0.50)
```

local_z以当前tile frame为基准。不能改为上次足端接触面、四足平均支撑面或机身出生高度。严格小于才触发，相等不触发；平台区z_expected=N*h。flat与楼梯前local_x<0不启用本项，其余bounds/body等安全条件仍照常。

当前events.py使用上述expected公式，保留；修正文档措辞。增加跨阶边界、平台、flat、局部/世界平移和等号/nextafter测试；下坠failure进入课程F。

## 3. 最终冻结测试：批准R完整扩张范围

三速.2/.4/.6各100例，全部用：local_x∈[-1.20,-.80]m、local_y∈[-.10,.10]m、yaw∈[-10°,10°]，按现有case构造协议独立均匀采样，不施加训练ramp。目标L40 N40/h.20/d.20；test seed固定400342，只按既定接口处理一次。

这不是“基础范围更严格”；基础范围更窄。验收应确认所有test case表明确标full_R_expanded，并与实际reset一致。只检查样本落在范围内不足以证明没有误用窄范围，要测试case生成器确实使用这组参数及已冻结表。

R两次课程门禁通过仅进入ready_for_final_test（非训练通过状态）；最终300例每档≥95才complete。最终测试有效但未达标记gate_failed，stop_reason=final_test_failed并保留各档计数；异常/缺例/未完成不能当成有效的最终失败计分，更不能complete。不得训练后补考或换模型。

## 4. 状态：批准八类训练停止名，补充完成与工程错误的语义

冻结以下训练停止状态与触发映射：

- behavior_regression：flat或H00独立保持集连续2次≥12/15失败。
- behavior_collapse：Q001明确的停滞诊断连续2次命中。
- curriculum_stalled：课程内部minimum_level_failure，即最低子级4个窗口F_EMA≥.80且满足原启用条件。
- budget_exhausted：总30000更新耗尽且无完整完成路径。
- gate_failed：子级/宏阶段预算耗尽且不能合法前进，或有效最终测试未达标；用stop_reason和budget_scope区别。
- user_requested_stop：用户停止，不算训练失败或监控失败。
- monitoring_blocked：真实监控投递/回读超过既有120s约束，不能用于用户停止。
- resource_insufficient：资源守卫未达标。

八类不是全部生命周期状态：running、ready_for_final_test、complete分别保留；complete只表示最终测试已通过，不能用于课程结束或worker退出。工程错误如非数值、评估执行错误、恢复合同不一致必须保留独立错误类别/证据，不能强行归入上述行为失败；已有类型化异常可继续使用，但必须由supervisor记录实际error_type，阻止继续训练且不伪造complete。

runner/spec/recovery/orchestration输出必须使用同一状态源并携带reason、global_update、phase、substage、generation、各层budget、触发证据。不得保留主路径把所有暂停或退出都映射为pilot_complete/failed的旧逻辑。pilot_not_applicable不是v06实际训练终态。

## 5. 预算优先级：有条件批准，总预算优先只用于预算分类

正常、可安全继续评估的完整更新边界：先保存本更新并执行当期应做门禁/诊断/保持集，再根据结果判断合法晋级或最终测试资格，最后判定预算。

同一更新同时到总预算与阶段预算且未具合法完成资格：主状态budget_exhausted；证据同时记录两个预算耗尽，不丢阶段失败事实。未耗总预算而阶段/子级耗尽、无合法前进：gate_failed，reason=phase_budget_exhausted或substage_budget_exhausted。单次门禁未过而预算仍有余量，不立刻gate_failed。

阶段最后一更新达到合法晋级，且总预算尚有余量：允许提交晋级，不先报预算失败。总更新30000恰好满足R最终晋级条件：允许零学习的最终300例测试；通过后complete，否则按§3记gate_failed或真实工程错误。不得第30001次学习。总预算30000只达到中间阶段晋级时仍budget_exhausted，因为后续训练未完成。

用户停止、非数值、资源不足或监控故障可以阻止继续评估，不得为完成门禁强行运行GPU；保持实际停止原因。behavior_regression/collapse/curriculum_stalled也不得被预算标签覆盖，全部触发事实可同时保存。先前已终止的run不重新分类。状态优先级实现测试须覆盖这些同时发生的情况，而不是只测试孤立布尔分支。

## 6. 保持集首次检查：批准严格下一个全局200倍数

分别记录flat、h00的首次qualified_global_update=q：

first_due = 200 * (floor(q / 200) + 1)
后续next_due每完成一次对应检查加200。

示例：q=200→400，q=350→400，q=400→600。同一更新刚首次取得资格不立即检查该保持集；两个集合都due时都执行，不能只返回其中一个。

“独立计时”表示两个集合各有qualified_at、next_due、last_completed、streak；不是从q起每隔200（例如350→550）。资格一旦取得保持开启，不因H退级或generation改变反复推迟首次检查；保持集失败计数不因另一个保持集通过或课程变化清零，只按本集有效通过清零。

checkpoint保存每集调度/结果身份，恢复按pending评估事务补完，不重复累计同一update结果、不无声跳过due评估。schema8内部字段变化在首次训练前重新冻结即可，不兼容旧未冻结工程容器冒充正式checkpoint。

## 7. V ramp：批准V阶段内累计已完成学习更新数

u=V阶段内已完成PPO更新数，进入V时u=0；p=min(max(u,0)/1000,1)。u=0区间[.3,.8]，u=500为[.25,.7]，u>=1000为[.2,.6]。不使用全局update，不因episode reset、evaluation、checkpoint或resume清零。

第1个V rollout开始前u=0；完成第1000个V更新后u=1000。episode仅在reset采新命令，已开始episode继续原采样值，不在rollout边界强制重采命令或额外reset所有环境。范围变化影响之后的采样，实际执行指令必须逐帧记录。

V完整门禁从第一天固定最终三速；只有完成V更新>=1000后的门禁计晋级streak，最早更新1200才晋级，仍保持SDS规定。R ramp同理用R内已完成更新数/400，不改变既有R规则。

## 验证与执行边界

必须增加上述边界与恢复CPU用例，尤其最后一更新合法通过、总预算末次进入最终测试、严格下一200倍数、两个保持集并存、V恢复不重置。

本轮发现runner仍有共享retention.streak/只返回一个集合的候选路径，及complete状态优先分支；这些在并行修正中，不把只读中间快照当最终缺陷结论，但验收必须确认主路径已满足本裁决。旧通过计数不替代修正后的同SHA完整CPU/独立复验。

Q001+Q002修正与完整CPU/独立复验通过前GPU仍阻塞；通过后按既有授权smoke→probe→独立最终工程验收→自动新v06 seed42。未授权换seed/加预算/重启R2 seed43。

主Codex本轮只读核对问题、events/runner/curriculum及旧下坠定义；执行调度/阈值/范围的独立算术断言，未运行生产测试或仿真，未验证训练效果。
