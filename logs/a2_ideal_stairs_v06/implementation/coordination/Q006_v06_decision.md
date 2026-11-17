# Q006 v0.6：元数据恢复缺陷、50次损失与一次性回退重放裁决

日期：2026-11-17；裁决者：主Codex。
问题SHA256：10d2d8adfa189df8cc3cf1497211834a1df6acee75498b127bd645e11d4056c1。
生效合同为原SDS + Q001–Q005 + 本补充；本文不修改训练配方。

## 1. 裁决：批准 A′，不接受 A 的成本口径，不批准 B
修复及下列验收通过后，批准从指定 boundary_000100.pt 回退重放，继续同一 trial02；不新建 trial03，不更换seed、不fresh初始化。此次是一次有损回退授权，不能再称“没有重放已计费更新”。
唯一新的执行段目录：logs/a2_ideal_stairs_v06/train_seed42_trial02_resume_000100_q006；必须不存在才创建。run身份仍trial02，恢复段身份独立。原trial02、Q005恢复段、所有失败目录/日志/control.json保留原字节。新目录不是新fresh run，模型不得来自工程容器。
原Q005恢复段control.json的resume_authorized=false保留为历史；本裁决仅授权在新段以受限恢复桥读取指定checkpoint，不得修改旧control或将用户停止的R2 seed43解禁。

## 2. 现场核实：实际丢失101–150，共50次
主Codex逐一读取commit_000101.json至commit_000150.json，共50份连续提交；最后一份committed=true、update=150、spent.F0=150。trace表明ppo.update成功后，commit_learning_update已执行，异常发生于finalize_boundary的save('boundary')，即第150次诊断前保存，而非100间隔的普通例行保存。
因此“仅101–149共49次”不成立。telemetry在边界完成后才写，缺少第150次telemetry不能反推PPO没执行。现有证据支持50次完整更新：1,228,800控制样本、4,915,200物理子步、1,000 Adam。DSH须追加问题勘误，引用这50份提交、冻结代码与trace；不能追改原问题或原计费证据。
恢复段checkpoints目录存在且为空。原boundary_000100.pt重新复算SHA仍为8ca0760e8c7a67da2c2143927e6ba5baf2459deba1967b0756a2aa601e58e805；CPU加载确认update100、Adam2000、2457600控制样本、9830400子步及initialization.transfer_metadata存在，源ETH SHA为8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722。
本轮不据工程异常判断奖励/课程成败，也不称update150诊断失败。

## 3. 预算：显式一次性增加实际计算额度50，不隐藏损失
- 活动模型分支逻辑预算仍30000更新，恢复起点100，最多剩29900；各阶段原预算不变。
- Q004的trial01损失50仍保留；本次trial02丢失段101–150再增加50。二者都是真实正式执行成本，不能通过改标签从总账剔除。
- 本次明确一次性批准正式执行累计上限由30050增至30100：trial01 50 + trial02最终活动分支最多30000 + 本次丢失分支50。满额对应739,737,600控制样本、2,958,950,400物理子步、602,000 Adam。smoke/probe/零学习评估另列。
- 当前已发生正式学习200次=trial01 50 + trial02原段100 + Q005恢复段50；这不是当前可恢复模型的update值。
- 新台账同时给effective_learning_update、physical_training_updates、lost_branch_updates、engineering_test_cost；按(run_id,segment_id,update)标识实际执行，不能因101–150编号重复就去重掉真实成本。活动分支内部不双计同一提交，外部总账如实累计。
- 新段元数据明确rollback_from_update=150、resume_from_update=100、lost_updates=50、replay_billed_updates=true（本次批准的有损回退语义）。不能复用Q005的false声明。旧台账不改，追加Q006版本。
- 未授权进一步损失补偿或新fresh；再出现不可恢复区间须重新裁决。

## 4. 修复必须证明恢复后的模型可再次保存
1. 从已校验checkpoint的initialization.transfer_metadata深拷贝恢复模型属性；缺失/非法/与ETH或schema冲突须在新学习前拒绝，不能默默跳过、编造默认metadata或重新执行ETH迁移覆盖已训练权重。
2. metadata是初始化来源证据，不是把已训练模型的融合层重新置零的指令。保留模型、Adam全部状态/关联、当前lr=8.888888888888888e-05、normalizer、RNG、课程/门禁/账本。原初始化历史与本次resume来源分别记录。
3. 新段load后、任何评估和学习前，立即原子保存完整resume_loaded_000100 checkpoint，再通过正式recovery验证并加载到独立裸模型/优化器进行一致性检查。验证不得改变活动模型或活动RNG。保存/验证失败即终止，不开始101。
4. 完成第100次pending评估和一次课程决策后，再保存并验证完整resume_ready_000100 checkpoint（pending已清、结果/streak/课程已提交），之后才允许101。这样已经完成的100门禁事务有可恢复模型证据，不再只留JSON。
5. 学习中仍按原100例行保存、50诊断、100门禁与所有到期评估前pending保存。不要本轮顺便改为每步保存或改变任何学习设置。
6. 新受限恢复桥绑定指定旧checkpoint完整SHA、新源码清单/闭包、Q006及历史Q005、diff、独立验收报告。保留旧模型来源，不关闭任意来源检查。由本次新代码合法保存的新checkpoint须能在相同冻结源码下正常二次恢复，不再误当原旧来源或要求伪造旧SHA。

## 5. update100门禁与有损回退的明确语义
Q005恢复段update100的60/60通过记录是有效历史，不能删除或称作无效数据；但没有持久化的post-boundary checkpoint可直接恢复完整边界状态。
本次限定从原pre-boundary100容器恢复，允许一次按原case表重建episode，补跑validation60例与diagnostic15例并完成课程决策，不执行PPO100、不计学习成本。这是工程恢复必需的重放，不允许重复评估直到通过。
在开跑前就固定：新活动分支采用这次完整结果；旧通过只留历史对照，不选较好结果、不拼接case、不叠加两次通过streak。若本次未通过，按原F0规则继续/停止，不因历史曾通过而覆盖结果，不因此再次回退抽签；若门禁或监控触发停止，照原规则执行。
case表身份、update/phase/generation和Q005固定值一致；评估必须隔离训练状态，正常保存恢复后的RNG。旧结果与新结果都链接同一模型SHA并标明恢复段，不宣称物理逐位复现。
CPU/GPU工程验证中的评估不能作为该次正式门禁结果。活动分支streak最多计这一次100门禁；最早仍需200另一次通过才晋级。若F0在200取得保持资格，首次保持检查为400。

## 6. 本次验收重点：实际恢复入口的往返完整性
不能只证明给测试模型手工挂metadata后save成功。必须覆盖与生产一致的裸V06ActorCritic→worker preload→runner.load→保存→正式recovery reload，真实非空Adam、原ETH来源字段、实际case_runner与受限桥。
CPU至少证明：
- 旧代码在裸模型load后save复现本错；新代码恢复metadata后可完整保存且正式加载，tensor权重/Adam/normalizer/lr与源一致；metadata深拷贝无共享引用。
- 缺失或伪造metadata在首次新学习前被拒绝，不能先做49次；原始错误不会被异常保存失败掩盖。
- 从真实100 pending状态补齐边界后保存ready状态，二次load不再重做已提交100任务/不双增streak，下一学习101。工程测试可隔离执行，不能手改学习计数冒充已运行。
- 真实runner组合覆盖150诊断前checkpoint和200门禁保存，写盘故障保留最近有效resume_loaded/ready状态；第150次即使telemetry未写仍计已完成PPO，不回滚真实消耗。
- 二次恢复来自新保存文件的来源检查可通过；不匹配源checkpoint/闭包拒绝；恢复后的保存与独立reload验证对活动模型及RNG无副作用。
完整CPU通过后，授权新冻结上一轮8env/H00/2更新smoke及一轮原56更新probe（5+50+1）；保持Q004/Q005真实50诊断与55→56要求。本轮probe必须经生产裸模型恢复入口，55加载后先保存并正式reload验证，执行56后保存再reload，不能沿用已有transfer_metadata的原对象假装重建。
独立D必须针对为何第9版漏检“恢复之后能否再保存”给出具体证据与修正测试；不能只给passed总数。检查学习逻辑未变、损失50与新累计30100正确、同SHA闭包和受限恢复桥闭合。GPU失败保留现场并上报，不无限重试；工程成本逐轮列出。
本轮无需为了等真实150再额外做150个GPU学习更新，边界组合回归与原56工程额度足够；不能消耗正式分支来验收。

## 7. 放行与回执
修复→完整CPU→新同SHA smoke/probe→独立验收blocking=0→追加冻结/受限恢复桥→自动新建上述Q006恢复段。无需再次申请继续，但生产首次resume_loaded及resume_ready保存验证必须在101之前成功；否则停机报告。
主对话协调、子agent实施、独立子agent验收。先回执本裁决SHA及50次勘误；启动后回送旧checkpoint/新闭包/验收/两个100 checkpoint的SHA、重新完成的100门禁结果、首次101提交与实际成本台账。不commit/push，不恢复R2 seed43。
主Codex本轮只读核查50份提交/trace/空目录，CPU加载原checkpoint与复算SHA/成本；未运行生产测试、GPU或仿真。修复与训练效果均待DSH实际验证。
