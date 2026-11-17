# Q007 v0.6：保持集地形路由修复与update400事务恢复裁决

日期：2026-11-17；裁决者：主Codex。问题SHA256：04f1d6fe211627b5c2cb027fa5ecfc97c52c080493f1397fce52eab22cbc0782。
本文补充SDS及Q001–Q006，只修保持集评估协议和指定恢复事务，不改训练配方。

## 1. 裁决：批准A′，纠正恢复源及case身份迁移
批准修复、完整CPU、新冻结smoke/probe、独立验收blocking=0后，继续同一trial02，唯一新执行段为logs/a2_ideal_stairs_v06/train_seed42_trial02_resume_000400_q007。不是fresh，不换seed，不重训任何已完成PPO。
指定恢复源改为Q006段的checkpoints/exception_000400.pt，SHA256 **50e939963e7986477f7f9472daa9e1bf527f11b08d11c67d957a1af7092492b5**。不要按原请求直接加载boundary再声称已跳过其并未保存的结果。
boundary_000400.pt（2a81110d588e5822eded1029f6a4180a6f78a6fa58c0231c65e360adaf4f3097）保留为评估前对照，不能更改字节或混同两者。若指定exception后续验收不合格，停机提问，不自动改选boundary、重跑门禁或拼接状态。

## 2. 主Codex已核实事实与必须勘误
- boundary400内部pending.results=[]，gate.records只有100/200/300，没有400；磁盘pending_boundary.json的完成状态不能倒推已写入先前checkpoint。
- exception400内部pending.results有400 validation及diagnostic两项，gate.records包含400通过，诊断历史也包含400；flat retention尚未提交，curriculum_decided=false。
- exception400的stop_state=None、lifecycle=running、pending_update=None、incomplete_update=None；CPU正式validate_checkpoint(num_envs=512)通过。故文件名exception不等于行为失败或不可恢复容器。
- 两checkpoint模型tensor、Adam、normalizer、learning、curriculum、初始化块逐项一致；学习400/Adam8000/控制样本9830400/子步39321600，保存lr=0.00011390625000000003（约1.1390625e-4），不能沿用第100次的lr。
- 两者Python/NumPy/Torch CPU/env/sampler RNG一致，但CUDA RNG不同。恢复保留exception自身完整RNG，不能从boundary拼接覆盖，也不得声称两容器全状态逐位相同或全部RNG隔离已证明。DSH独立审核须记录这一差异及来源调查；若进一步发现违反既定评估隔离合同，单列问题上报，不在本次静默改随机采样实现。
- 源码核查证实原retention取_current_tile；主Codex核查期间DSH已并行修复，后续CPU调用已能拒绝错误tile，并使H/V/R保持集用初期速度及standard出生范围。这只是候选纯函数核验，不代表冻结生产验收完成。
- 请追加Q007勘误，不覆盖原问题。H00是初级楼梯，300/400通过不能称40级20cm目标楼梯已通过；flat保持结果尚未知。

## 3. 固定保持协议：与训练阶段分开
1. flat_retention固定spec.FLAT_TILE_ID=0；stair_retention固定tile_id_for('H00')=1（6级、h=0.02m、d=0.30m）。target_tile在普通phase规则之前识别保持集。显式传入错误tile必须拒绝；不应悄悄忽略调用方错误。
2. _due_tasks、run_gate默认/显式入口、make_cases/check_table、实际env reset与pending重放都使用同一个保持目标。正常validation/diagnostic仍评当前任务，不能因修保持集让H/D/L/V/R普通门禁都落到H00。
3. 按SDS§6.3，两种保持集始终三档初期速度0.3/0.5/0.8m/s，各5例、共15例；seed42用原保持集seed映射（validation offset仅加一次：200342）。固定standard出生x[-1.1,-0.9]m、y±0.05m、yaw±5°，沿用原随机生成顺序。V/R的普通门禁速度/扩张初态不传入保持集。
4. flat按F0500步、真实同episode净位移>=5c-1e-9（1.5/2.5/4.0m）、max|y|<=0.20m、max|yaw|<=10°及原失败条件；Q005一致性容差1e-6m、rtol0仅用于证据一致性。stair按H00原登顶/失败/超时公式，不误用500步。
5. phase/substage/generation可保留当前训练上下文用于计费与溯源，但不能改变测量地形、速度、case姿态或判据。固定case的物理内容跨阶段一致；如完整表hash包含上下文，允许其不同，但必须证明实际case内容未变。
6. 保持两个集合独立时钟和失败streak；每集连续两次至少12/15失败才behavior_regression。错误地形异常不算一次真实失败，也不能算通过。本次完成flat400后next_due=600；若本次课程决策确认H00双门禁资格，qualified_at=400、首次H00保持=600，不追补400的H00保持；600同时到期需各执行一次。

## 4. 指定pending任务的一次性受限迁移
即使选择exception400，其中尚未完成flat任务仍写tile1和错误case表SHA；只修target_tile无法直接重放。授权恢复桥针对这个唯一源SHA做明确迁移，不能关闭Q006 case表身份校验。
- 源pending必须为update400/phaseH/substageNone/generation1，committed=true、curriculum_decided=false；已完成两项身份必须匹配源容器与磁盘证据。
- validation SHA **0ccaa01f4161fa1006d49e41cb8bec298e68bd6c76a7f3d9bb195de603e45fd7**、diagnostic SHA **d7468056efc6ef8911fa26736a430f16a2d68b4717c22640a150e146e1b6caf8**保持原样；继承其gate/diagnostic记录和streak，不再次调用评估、不再次应用streak。
- 只替换未完成flat retention：旧tile1/旧表SHA **afb64ce64395297bd73ea3a842b1c8a9ecc1ae7fce97d0fe888cbf60cfecc5eb** → tile0/新固定协议表SHA。其旧失败/部分case证据标protocol_invalid并保留，不拼接新表，不把旧表标成新表完成。
- 新表由冻结代码按原seed/初期速度/出生规则生成；在执行前将完整旧任务、新任务、新表内容及SHA、替换理由、Q007SHA写入追加的迁移清单。地形变了就必须有新身份，不能沿用旧hash。
- 恢复桥验证精确源SHA/旧任务/旧结果/新代码闭包/独立验收；只有这一未完成任务可迁移。其他任务、未知hash、已完成错误任务、源模型/计数不符一律拒绝。对已迁移新checkpoint二次load不得再次替换或双计结果。
- 不修改旧checkpoint。把受限迁移后的新内存状态保存到新段resume_loaded_000400，正式reload一致性验证成功后，才执行一次flat15例。随后一次课程决策、一次原有晋退/预算/停机处理，保存验证resume_ready_000400，再允许401。
- 若真实保持结果/其他规则要求停止，按原合同保存并停止，不为进入401重评。若只发生普通首次保持失败且不达停止条件，保留其失败streak并按原规则继续。

## 5. 成本与恢复身份
rollback_from_update=400、resume_from_update=400、lost_updates=0、replay_billed_updates=false、replay_pending_evaluations=true。最后一项仅指未完成且经迁移的flat保持任务；已完成门禁/诊断不重放。
活动分支cap30000，已消费400、剩29600；累计正式计算cap仍30100，无新增损失。正确公式为trial01损失50 + Q005丢失分支50 + trial02活动模型分支最多30000；原trial02前100已包含在活动分支中，不能另加一次。当前实际学习累计500=活动分支400+两项损失各50。工程smoke/probe与零学习评估另列。
继承Q006所有恢复保存验证要求：真实裸模型重建、完整Adam/metadata/normalizer/RNG及当前lr、受限跨源码来源、loaded与ready保存验证、来源不明拒绝。旧Q006段、control.json与所有原证据保留，不恢复R2seed43。

## 6. 必须覆盖本次真实路径的验证
- 完整CPU：F0/H/D/L/V/R × flat/stair retention矩阵，目标0/1、三速各5、固定姿态、错误显式tile拒绝；普通validation/diagnostic原路由不变；实际runner到env reset使用同一目标；8env批量与末批不足8、500步/提前失败/auto-reset/端点不跨episode。
- 真实400源容器集成：exception通过正式schema与受限桥；boundary作为不含完成结果的负例不得混用；只迁移flat任务，validation/diagnostic调用次数0、flat执行一次；计费400/Adam8000不变；一次课程决策，正确更新保持时钟；ready二次load不重跑已完成任务，下一学习401。验证50/150式保存故障路径仍受保护。
- 600双保持集故障注入：完成flat后stair异常再恢复，不重做flat/不双计streak；已完成结果必须来自持久化checkpoint或受验证事务证据，不能像本次误把外部JSON当作旧boundary内容。
- 同新冻结上一轮8env/H00/2更新smoke与原56更新probe（512训练+8评估、5+50+1），保留50诊断及55/56裸模型恢复loaded/ready验证。另在工程隔离评估中实际执行两个保持集各15例、目标分别flat/H00，使用同一冻结学习模型、零学习，检查真实8env批量目标/步数/完整记录。工程策略通过率不设放行门槛，协议正确且无工程异常才是工程验收；不允许拿它替代正式400保持结果。不为凑400/600额外做GPU学习。
- 独立D给出为何旧验收遗漏“训练phase×保持kind”组合、为何boundary与exception完成状态被混淆、如何验证迁移后hash及去重。不得仅报passed总数。记录CUDA RNG差异调查，不声称本轮已证明全部随机状态隔离。
一轮工程失败则保留证据上报，不无限重试；CPU/GPU/独立验收绑定同一最终源码。以上保持协议修正落实SDS原设计，不是放宽门禁或修改训练配方。

## 7. 放行
按修复→完整CPU→新冻结smoke/probe及零学习保持验证→独立blocking=0→新受限桥清单→自动Q007段恢复。无需再次询问是否开训；未满足不得启动正式段。主对话协调、子agent实施、独立子agent验收。
先回执本裁决SHA、恢复源更正及任务迁移方案；完成后回送exception源SHA、新闭包/验收/桥SHA、旧→新flat表SHA、loaded/ready400 SHA、正式flat15例结果、完成项去重证据、课程/保持时钟及首个401提交/成本。无需保证保持通过或401必达。
主Codex本轮只读源码/文件与CPU checkpoint/schema校验、状态比较及候选纯函数检查，未运行GPU/仿真/完整生产测试。未改生产代码，修复及保持效果尚待DSH验证。不commit/push。
