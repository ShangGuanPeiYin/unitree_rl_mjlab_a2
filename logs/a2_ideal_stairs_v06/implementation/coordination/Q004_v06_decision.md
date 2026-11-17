# Q004 v0.6：trial01工程崩溃与一次性重开裁决

日期：2026-11-16；裁决者：主Codex。
问题SHA256：6e114894ea253ec31cbf6000a5d128cb1342d173abaa646aab2261cd0bafeebb。
生效基础为原SDS与Q001/Q002/Q003，本裁决仅对工程恢复、验收和一次fresh路径例外作补充；不改变行为配方。

## 1. 裁决：批准A，附本文件的修复和验收条件

批准修复→完整CPU→新源码冻结及GPU smoke/probe→独立最终验收blocking=0→新建train_seed42_trial02。不是立即训练授权，必须先完成以下验收。继续seed42、ETH fresh初始化、原奖励/PPO/课程/门禁/初态/速度，不挑选历史模型，不复用任何工程检查模型。

Q001-5固定路径增加一次例外：本次唯一允许的fresh目标为logs/a2_ideal_stairs_v06/train_seed42_trial02。trial01永久保留为工程中断证据；不得清空、覆盖、重命名复用或从无效文件伪造resume。trial02若已存在即拒绝fresh；任何未来trial03或再次fresh重开需另提裁决，不形成可无限重跑的授权。

trial01在学习update50的诊断路径发生实现异常，本轮不据此判定奖励/课程成败。不是“训练效果不合格后换一次抽签”。主Codex读取时checkpoints目录存在但为空，纠正问题正文“目录不存在”的细微描述；关键事实是无可恢复checkpoint。

## 2. 成本明确记账：不能称已计费50被抹去

trial01模型状态无法恢复，失去的是可继续学习的权重状态，不是已发生的计算成本。

trial01记录：50更新；1,228,800控制样本；4,915,200物理子步；1,000 Adam更新，最终以实际日志逐项复算并标出无法证明的字段。

本次明确批准替代run仍有原单run上限30,000更新及原各阶段预算。这是一项一次性的工程重开成本例外：两个正式尝试累计上限为30,050更新（不含单列工程检查），不是项目累计仍为30,000。不得把trial01的50加到trial02内部学习计数，也不得假装trial02从update50继续；trial02从0开始。

若trial02用满，两个尝试累计738,508,800控制样本、2,954,035,200物理子步、601,000 Adam更新；smoke/probe及历史工程失败另列，不隐藏。新建attempts台账，将trial01标implementation_crash_nonresumable，引用异常/代码冻结/验收历史及证据SHA。旧原始记录不改写。

## 3. 修复一：真实非空Adam状态的完整快照

- optimizer.state_dict()['state']按parameter_id→per-parameter-state映射遍历，使用.items()而非enumerate(dict)。param_groups仍按列表顺序处理。
- 摘要必须覆盖参数ID关联、状态tensor（step/exp_avg/exp_avg_sq等）、非tensor标量/结构，以及param_groups内lr/betas/eps/weight_decay/params关联。不能只修异常后继续忽略非tensor状态；不允许把异常当“状态未改变”。
- 使用确定性、保留类型/shape/dtype/值与映射身份的规范化摘要。映射排序可规范化，但不能依靠value拼接丢掉参数ID，不能将NaN转成null后默认为有效。
- 读取快照不得修改优化器、模型、normalizer、RNG或将GPU tensor留在长期日志中。原评估状态隔离要求全部继续有效。

验收至少：真实torch.optim.Adam执行一次及多次step，产生非空state后，快照与真实evaluate_checkpoint路径不崩溃；无变动摘要相同；分别改exp_avg、exp_avg_sq、step、lr、参数ID对应关系、非tensor状态，均被识别；空Adam也能处理。不得只用fake optimizer或未step的空state测试。

## 4. 修复二：学习计费提交与课程决策拆开

不能仅交换一次on_update与评估顺序，因为on_update当前还包含晋级、退级、EMA处理与预算判定，提前调用会违反Q002“末次先评估再决策”。必须明确两阶段：

A. commit_learning_update(u)：PPO完整更新成功后，先把本次学习成本记入旧phase/substage/generation。更新learning、last_update、phase/substage累计、generation_updates等，满足sum(spent_by_phase)==learning.update。只计费，不晋退、不判预算失败、不消费第二遍课程窗口；对相同update重复提交必须幂等或拒绝，不能重复计费。

B. finalize_boundary(u)：在已计费的旧阶段模型上完成到期门禁、诊断、保持集，之后按既有优先级消费一次课程窗口、晋退/预算/完成判定，并提交边界事务。恢复时不得重做PPO或再计费u。

必须记录pending_boundary：update、旧phase/substage/generation、到期任务列表/每项case表身份、已提交结果ID、尚未完成步骤、是否已执行课程决策。评估结果以(update,kind,retention_kind,generation,case_table_hash)去重，不能恢复后双增streak。不得调用无门禁on_update先判fail，再补写通过结果。

新增持久化要求（明确的工程修复，不改学习配方）：
1. fresh初始化完成、首次学习前保存有效update0 checkpoint；必须在任何基线评估等可能失败的外部工作前完成。
2. 每个到期评估边界（包括50诊断、100门禁、200保持集）在调用评估前，保存已计费、带pending_boundary的完整checkpoint。原100更新例行保存和阶段转换保存继续保留。不要只靠except临时补救。
3. 评估异常时保留原始错误和最近有效checkpoint路径/SHA。异常快照保存若再次失败，另记checkpoint_save_error，不覆盖原异常、不把损坏临时文件当checkpoint。原子写入、校验和/完成标志及rename按现有恢复安全要求实现。
4. 不允许通过关闭validate_curriculum、放宽账本不等式、只保存模型不保存Adam，或伪造spent/update来让保存通过。
5. 如异常发生在PPO中途而非完整更新，不能把部分Adam步骤伪装成一个完整更新。保留实际已完成的Adam计数与最后有效checkpoint，状态标明不可直接续接的区间，提交裁决；不要自动回退重放已计费部分。

完整更新后的评估异常可从pending checkpoint重建新episode仿真，先完成待评估事务/课程决策，再进入下一学习更新；不声称恢复原物理轨迹逐位连续。用户停止/行为失败不得因有checkpoint而自动恢复；本次trial01仍无checkpoint，不适用此路径。

## 5. 必须覆盖本次真实路径的验收

### CPU与故障注入
- 非空真实Adam快照测试，见§3。
- 在update50的诊断执行中主动抛异常：checkpoint应含learning.update=50、对应累计spent=50、1,000 Adam、1,228,800控制样本（512×48配置）、pending诊断；通过正式recovery验证并可加载，不仅检查文件存在。
- 从该checkpoint恢复：先完成未完诊断/边界，不重训update50、不重复计费/结果提交；下一次学习是51。
- 在100门禁、200双保持集、阶段最后一更新/总预算30000处故障注入，验证原Q002预算顺序与两个保持集独立去重。
- PPO半途异常与checkpoint写入异常：原异常不被二次异常掩盖，不能标complete或静默继续。
- 用实际runner→evaluation→recovery组件组合测试，不能仅mock掉diagnostic/save/optimizer.state_dict后宣布通过。

### GPU重新验收
- 在新冻结源码上重做8env、H00、2更新smoke。
- 重做原56学习更新probe（5预热+50测量+1恢复，512训练+8评估，成本单列），必须通过实际runner的第50更新边界、真实非空Adam快照、真实15例诊断、pending checkpoint保存和加载；最后一次恢复更新验证从55到56。probe可保留SDS的最终地形资源场景，但其工程账本、generation及实际场景必须自洽，不用F0虚假字段掩盖R场景。工程容器永远不可作为trial02初始化或恢复源。
- 保持原四项资源限制及接线，不通过变更资源阈值/禁用诊断来跑通。
- 本次授权新的完整smoke/probe各一轮作为修复验收；若又失败，保留证据并上报，不能把无限重复试跑说成同一轮。

独立验收D必须补充“旧验收为何漏过”：原快照测试是否只覆盖空optimizer，原probe是否未通过真实runner诊断边界，以及异常保存事务未覆盖何处；逐项给新测试及证据。不得只增加总passed数，必须证明这两个崩溃链已被测试覆盖。

## 6. 冻结与放行

追加新代码闭包、SDS+Q001..Q004补充合同、路径授权和证据manifest；原验收报告保留为历史，不追改其SHA或声称它覆盖新源码。状态schema8在正式trial02开始前冻结新pending事务字段，老工程容器按身份拒绝或明确处理，不静默补字段。

新同SHA完整CPU通过、smoke/probe通过、独立最终验收blocking=0后，DSH按本一次授权自动启动trial02，不必再问是否训练。请先回执本裁决SHA、修复分工、拟增加的关键回归测试；启动前再回送最终冻结SHA、验收SHA、trial01成本台账与GPU证据路径。

实施子agent与独立验收子agent仍分离；主对话协调。未覆盖问题继续提问本Codex，不自行推测。不得commit/push，不恢复R2 seed43。

主Codex本轮仅核查问题SHA、实际trace、空checkpoint目录及并行修改中的快照/runner/recovery代码，未运行生产CPU/GPU或训练。无可恢复模型的结论不等于评价此次50更新的策略效果。
