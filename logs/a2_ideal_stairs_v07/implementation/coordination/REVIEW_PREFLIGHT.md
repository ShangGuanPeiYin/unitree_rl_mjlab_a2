# v0.7 独立接口预审

状态：预审已完成；实现和最终冻结验收尚未完成。本文件不是accepted=true/blocking=0放行报告，不能据此开始正式学习。

## 身份、授权及读取范围

Reviewer=/root/c1_reviewer，沿用本agent原显式派工gpt-6.1-sol/high配置；主协调/root，A=/root/v07_terrain，B=/root/c1_executor。以实际派工记录为依据，不冒称额外底层模型运行证明。本reviewer不实现生产或执行者测试、不验自己实现模块、不派子agent、不启动GPU/训练、不Git写入。全局AGENTS已读；项目及dog中间目录无额外AGENTS。

完整读取603行SDS及GLOBAL_CONSTRAINTS、REVIEW/A/B briefs、执行提示和交接配置记录。SDS实际SHA256已独立重算：303ae40a5c5e3fe82f9f0df3aae8d153136e016439087c8db9701aeb17197aca。用户授权原始4506/4552由主协调专门核验；reviewer读到的是该核验交接，不把文档冒充原用户消息。当前只新增本审计文件，codex_changes由root统一追加。

## 唯一来源与新合同

唯一来源stop_004817.pt实际SHA已重算匹配a41963a8bf5e6f1d4810b0a0c912ac65747d61bd481ea3432ef638a99e076e64。来源是D/D25、gen72、update4817、Adam96340、control118382592、physics473530368，真实lr7.593750000000002e-05。迁移要比对完整model/Adam参数ID关联与param_groups/normalizer/ETH原证据，不能仅比tensor值或把原initial_lr3e-4覆盖到当前Adam。探索std和已学route/fusion不得重置。

v07必须原生code0.7.0/recipe0.7-d22-eval-regression-r1/schema9、architecture仍v06-route-state-v1，不能先激活C1再覆常量；加载旧metadata只允许明确的来源证据兼容，不篡改其原recipe/code。新spec闭包覆盖实际v07脚本及所复用公共依赖；旧v06/C1的97源、28测试及原4817证据保持字节。

只迁移这个精确源一次：新gen73、generation_updates0、当前EMA/FIFO/资格/hard/minimum streak/退级队列清空；legacy pending_transition原值写父证据，活动态清除。保留原学习费、已完成阶段、checkpoint自身RNG/sampler、flat/H00资格（4800完成/5000到期/streak0）。不能从result拼sampler。首新PPO4818应使Adam96360/control118407168/physics473628672；这些是合同算术，尚非已发生训练。

## A/B接口和物理编号

已收到A确认：CURRICULUM_TILE_ORDER=tuple(range(1,18))+(22,18,19,20,21)，不含flat0；PHYSICAL_TILE_COUNT23，FINAL_TILE_ID21。A所有spec/geometry/env文件与B的role_tiles共享此接口，handoff后按owner单写。

|训练课程|物理frontier|near物理tile|easy课程上界|
|---|---:|---|---|
|D25|17|15、16（H14/H15）|H13/tile14|
|D22|22|16、17（H15/D25）|H14/tile15|
|D20|18|17、22（D25/D22）|H15/tile16|
|L12|19|22、18（D22/D20）|D25/tile17|
|V/R|21|19、20（L12/L20）|D20/tile18，包含D22/tile22|

near/easy按难度列表位置，绝不能按物理ID±偏移或物理ID最大值排序。23个物理tile数组/接触/边界合法性不能以<=FINAL21判定。旧0..21应完整TerrainSpec和world-origin逐项不变；追加22原点(0,176,0)。D22统一TerrainSpec派生：6级、h=.20/d=.22、width.80、thickness.01、top2.0；platform x1.32/z1.20，机身成功线2.12；timeout .3/.5/.6锚点1828/1297/1164。不能用新增tile22作最终目标，最终仍L40/tile21。

## 保护期、正式评估及预算

D/L六子级保护期必须读当前generation_updates，而非D宏累计/阶段stage_iteration/64窗数。初迁移宏D累计2117仍保留，但gen73从0。首个全局门禁4900仅83代内更新，不能计晋级；若持续D25，5000/5100可累计两次有效正门禁并提前晋级（不被1000保护阻拦）。若一直未晋级，5800仅983不得入退级队列，最早合格坏评估全局5900/6000/6100，对应1083/1183/1283。中断/二次load不能重新获得保护期或抹队列。

D/L训练EMA及原64窗保留诊断但必须失去退级和最低停机决策作用；H原内部升降/最低规则保持，不通过改公共常量意外影响H。新D20回退D22、D22回D25、L40回L20、L20回L12；D25/L12最低只按新三评估条件停，不能并存旧F_EMA路径。

退级输入要直接来自本次已持久化完整validation：当前model/optimizer/update/phase/substage/generation、三速各20共60、first-record/table/summary三SHA均校验。不得从日志重查、复用历史passed bool、把诊断/保持或缺例异常当0成功。S<=12为低成功（12/13边界）；队列仅最近三个同代相邻100更新的正式低成功；S>12打断，重复事务不再消费，错序/冲突身份不能静默替换。Δ>=6仅continue，按“最近三次”滚动观察，不自行额外清队列或延长保护；Δ<6才退/最低停。gap处理与拒绝非法恢复队列应由B报告具体一致实现，不能使非邻接三条触发。

同边界顺序必须保持：PPO计费提交→到期评估与证据→原独立停止条件→晋级优先/未晋级再新退级→预算；保护不屏蔽保持、停滞诊断、资源/数值/用户停止或预算。必须覆盖同边界同时正门禁/保持失败/预算界的反例，避免晋级提前返回吞停止。

预算累计：D25/D22/D20为6000/6000/9000，D合计21000，全分支48000、含原100损失正式cap48100。源D25=2083/D20=34、D22=0保留；D余额18883、账面global43183，但F0/H未用额不转移，今后各阶段最多新增32883，不能把43183当保证可训练量。预算不足不自动扩容、回退不清子级费用。

## 两个必须闭合的工程事务接口

1. **每次64窗口的原行持久化与消费。**记录确切64行（或不可变原件索引/hash）、generation、消费update、排序键、role/tile/result/reason；原行证据必须在FIFO pop/EMA应用前或与可恢复pending原子关联。证据写失败后，异常checkpoint仍须保存该实际窗口与已消费/未消费状态，恢复先完成同一个boundary，不重新学习、不遗失64行、不重复EMA/计费。只保聚合/剩余FIFO不满足SDS5.4。窗口计数必须区分已消费与剩余，不以凑满候选样本混算。
2. **退级队列消费与晋退checkpoint的幂等。**evaluation_transaction缓存的实际测量身份属于旧phase/substage/gen；课程切换后状态属于新gen。必须有持久的单次决策/提交边界，将同结果sidecar、主记录、队列、generation切换和checkpoint关联；保存失败后不能拿新gen拒绝合法旧待完成任务、也不能重评/双计坏评估/重复退级。application_incomplete等不可证明完整的状态留证并拒自动恢复；不要为测试放宽普通终态resume。last-applied去重键至少包含SDS列明update/gen/phase/substage/kind/tableSHA，并绑定完整实际评估证据。

以上均为SDS既定不重复/证据要求的工程接线，不是新配方。已直接同步B，等待其模块报告/故障注入结果后独立审阅；不由reviewer实现产品。

## 已澄清恢复初始化 Qv07-001

提出的恢复初始化语义已由主决策明确裁定，不再悬而未决。补充文件coordination/V07_RECOVERY_INITIALIZATION_RULING.md实际SHA1f84b142b70cc952eeca6e0a630f76d5d28f3d3b542cf2bcf5084d19be94a3cf，需与原SDS一起进入最终冻结绑定，原SDS不改。

沿用恢复checkpoint完整RNG→原schedule一次→原ppo.reset一次；变化槽可能内部初始化后又全量new_episode，额外随机消耗已明确接受，不加seed、不回卷/拼接、不声称物理位精确。首/二load记录恢复前/后/schedule后/final reset后四时点六RNG及独立课程sampler、reset槽/次数、最终观测/奖励/终止/history/map缓存。首学习前D25/gen73/stage_iteration2117/512角色102/164/123/123、合法tile集合与command范围正确，训练不传tile_override。临时reset不计真实完成episode、课程统计或学习。原延迟奖励reset ids队列若沿用须说明首次初始化幂等清零，不能因容器非空误报旧episode统计，也不能漏查真实副作用。

## 独立验收清单与当前未测项

|阶段|必须取得的证据|本次状态|
|---|---|---|
|A native/geometry|23tiles、旧0..21等价、D22 collision/query/map/contact/timeout/success及课程采样|只读接口预审；A已写RED，未将其声称最终通过|
|B regression|999/1000边界、12/13、Δ5/6、rolling/间隔/身份/重复、保护不阻正晋级、DL旧EMA不决策/H原行为|等待实现diff+报告；reviewer不改tests|
|恢复/事务|精确4817拒绝反例、非空Adam全结构保留、首/二load、保护/队列/时钟保持、IO各边界故障恢复、64原行|等待代码与原始RED/GREEN日志，不复用旧C1通过当v07通过|
|旧行为隔离|原97源/28测试/4817原件SHA不变；native导入不污染旧模块，含外部C1 env-var存在时也不依赖其激活|root已before-freeze；最终再核|
|最终完整CPU|同一冻结source/contract/test清单，0 failed/0意外skip/0errors，完整原stdout与exit|尚未执行；不重复已过同SHA全套|
|最终GPU|串行D22 smoke8env2、512+8 probe56含50真实诊断/55→56、零学习512双load、D25/D22/D20三速45baseline、flat/H00各15含真实500窗|尚未执行，不以mock或旧GPU替代|
|最终成本/资源|PPO与评估有效/actual/extra分账；整卡7168/进程5120/host4096/disk10门槛，按样本说明非连续峰值|尚未执行；不缩并行/网络来过资源门槛|
|最终放行|独立MD/JSON精确绑定SDS+裁决+source/contract/tests及实际证据，accepted=true/blocking0后根预检/自动启动4818|本预审不放行、不重复询问已授权步骤|

没有基线成功率额外准入线，也不允许重跑挑好结果。新D22是否增加D20连续学习量、爬楼成功率是否改善，须由正式课程和最终300例证明；工程准备不等于行为目标完成。

## 本批记录

修改日期：2026-11-17T20:34:56.307561+08:00。目的：在产品实现前独立明确接口、状态边界和证据等级。涉及文件仅本REVIEW_PREFLIGHT.md；未修改生产、执行者tests、日志、原始证据。设计理由：将物理编号与课程排序、generation资格与累计计费、评估结果与64训练窗口、实际数据与推定分开，及早暴露持久化/跨代恢复风险。验证仅完整文档阅读及SDS/唯一源/裁决实际SHA重算，未运行测试或GPU。共享logs/codex_changes.md由root按本短报统一追加。
