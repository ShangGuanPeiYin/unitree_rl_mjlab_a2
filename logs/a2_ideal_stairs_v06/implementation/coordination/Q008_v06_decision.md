# Q008 v0.6：保持集记录语义与评估RNG隔离限制裁决

日期：2026-11-17；裁决者：主Codex。问题SHA256：907f3536fd6b96b141f449e7bc15df05bb28bf6a17171b0c57cecc72f6e28cf2。
本文补充SDS与Q001–Q007。裁决为A′：当前run继续，记录解释现在补，生产代码修正留到当前run停止后；O-15只作为明确的合同偏差接受，不视为已证明无影响。

## 1. 当前执行：不暂停、不回退、不新建段
允许当前train_seed42_trial02_resume_000400_q007按既有冻结源码继续到600双保持及后续课程，直到原有完成/停止/预算条件；不为了本次记录问题重跑400，不新建fresh或新的恢复段，不修改运行中的源码/模型/随机状态/门禁/预算/控制文件。
独立验收D继续进行下述只读审计与离线记录复算，不新增并行GPU worker。无需为Q008重跑smoke/probe，因为本裁决没有授权替换活动源码。
本次许可仅限当前trial02及当前冻结实现，不自动推广到下一个训练版本或其他run。代码替换与未来跨源码恢复仍须另行冻结验收，不能以本裁决为热更新许可。

## 2. 保持集passed为何不可信，以及可采信的事实
主Codex读取代码及flat_retention_000400.json确认：每速case_pass_count=5/total5、failed_cases=0、gate_eligible=false，但required=18、gate_pass=false、gate.passed=false。
这里有两层原因：required沿用普通门禁18，且gate_pass本身被gate_eligible限制为仅validation/test可true。只把18改成其他阈值并不能修复GateRecord.passed的语义。
当前实际路径：run_gate只在gate_eligible为true时更新晋级streak；保持集走_apply_retention→record_retention，停止判断只使用failed_cases>=12及独立连续计数，不以passed驱动。当前检查支持本次错误标志没有改变该路径的课程晋级和保持停机；不是证明任意未来消费者都安全。
400的正确解释是三速各5/5，共15/15通过、failed_cases=0、retention_alarm=false、该保持集失败streak=0。不能对用户报告成保持失败；也不能把400保持集当成另一次60例晋级门禁。
下一次600两个保持集各15例独立判定：单集失败>=12仅增加该集连续失败计数；连续两次才behavior_regression。0–11失败会清零该集计数。例：12→0→12→12只在最后一次达到2而停止。普通门禁18/20与最终95/100完全不变。

## 3. 现在就做的无侵入记录补充
不得等训练结束才纠正用户看到的解释。授权在coordination或独立audit目录追加Q008_RETENTION_INTERPRETATION.md及retention_interpretation_q008.jsonl，原gates/retention/checkpoint/W&B原始历史不覆盖。
每条解释绑定run_id/segment_id/update/kind、源文件路径与SHA、case_table_sha、三速各自分子/分母、总15、failed_cases、原raw_passed和raw_required、retention_alarm=(failed_cases>=12)、retention_threshold_satisfied=(failed_cases<12)、独立streak_before/after与stop应有值、实测值、匹配结果。
只有15例完整有效且无重复/缺失的记录才可复算；不能把缺失/异常算通过。优先用逐例case判据复算，若仅有汇总须标summary_only及证据限制，不能假称逐例已核验。
保持标志名“未触发退化阈值”不等同“15例全部成功”：例如failed=11仍threshold_satisfied=true，但只是未达到严重退化报警线，不能包装成良好性能。
D在当前run继续期间复核消费者：runner/课程/恢复/监督器/最终汇总/telemetry，确认无其他地方按保持passed直接触发晋退、停机、源checkpoint筛选或最终验收。记录该判断所覆盖的冻结文件SHA。后续保持完成后追加解释，至少明确回送600双集合计数及streak，不改生产遥测接线。

## 4. O-15：接受当前run带已知限制继续，不宣称隔离无影响
SDS§8.3要求模型、normalizer、optimizer、RNG、环境缓存和计数隔离。当前training_state_snapshot仅覆盖其中部分RNG：Torch CPU及不完整NumPy状态摘要，缺Python/CUDA/训练env/sampler及环境缓存的完整核验。现有statistics_unchanged=true只能解释为“当前已实现摘要覆盖的字段未发现变化”。
已知事实：Q006段boundary与评估后容器CUDA RNG blob有差异；训练Normal.sample和PPO torch.randperm(device=cuda)使用CUDA全局流，故不能直接断言变化与后续学习无关。另一方面，两个checkpoint之间存在评估、保存等操作，仅据两个端点不能确定具体调用来源，也不能证明奖励失效、策略损坏或评估分数错误。
本裁决明确接受本trial02暂时偏离“完整RNG隔离已验证”要求，继续收集爬楼学习效果；该项状态固定为accepted_with_known_limitation_Q008 / full_evaluation_rng_isolation_not_established。不能改写历史验收为完全符合SDS，不能用最终登顶通过倒推O-15不存在，也不自动宣布旧行为门禁无效。
保留当前随机流，不拼接旧checkpoint RNG、不重置seed、不在运行中增加fork_rng/保存恢复逻辑。这是避免中途改变活动训练执行方式的决策，不是对缺口无害的保证。
D现在做只读离线来源调查：区分训练/评估各自Generator、全局CUDA流、模型重建/验证save-reload与评估路径的随机调用，列证据及未定位部分。只比较checkpoint blob不能命名为“精确定位评估算子”。缺GPU能力时如实写未验证，不能用CPU或mock替代真实CUDA结论。发现改变训练参数/normalizer/optimizer/课程/训练环境缓存、污染case/训练数据、或错误标志实际驱动决策等新的实质证据，立即上报并在可保存的完整边界暂停保全；不回退丢弃更新，不自动重启。
仅存在本次已知CUDA差异与覆盖不足时，不再次自动暂停；这是本次显式接受的限制。真实保持退化/预算/监控等原停止条件照常生效，不以Q008豁免。

## 5. 当前run停止后的修复规格（本轮不在活动树实施）
先输出离线候选diff/测试方案及D审计报告，不在活动源码写修改。当前run停止后按此规格形成待验收修复，不自动获准启动新训练或跨源码续跑。
保持集记录必须显式分流：gate_eligible保持false；增加retention_alarm、retention_threshold_satisfied、fail_required=12、cases=15；每速case_pass_required=None、per_speed.passed=None，仅保留成功数/总数。若为兼容GateRecord需要bool passed，明确令保持集passed=retention_threshold_satisfied，并标verdict_type=retention_threshold；普通validation/test仍使用原gate_pass。不能为了让保持passed为true把gate_eligible改成true或参与晋级。
required=None不能偷偷改成每速12：12是总15例的失败阈值。门禁摘要到runner/telemetry/恢复反序列化必须一致解释，不能只修summary而GateRecord仍强制false。
测试覆盖六阶段×两保持kind×failed=0/11/12/15、连续计数/双集合互不影响、逐例到摘要/日志的端到端；普通18/20和最终95/100仍不变。纠正显示不改变原failed_cases、案例表或历史真实停止结论；历史用sidecar解释，不改原字节。
RNG修复另列设计：比较Python完整状态、NumPy完整状态、Torch CPU、已使用CUDA设备、训练env Generator、训练terrain sampler及必要训练缓存/计数；明确评估env自身状态可正常推进，训练env不可被误当评估env。验证模型模式和训练normalizer/optimizer不变。若要保存恢复随机流，必须异常路径finally恢复且有真实CUDA定位证据；不能盲目重置种子掩盖差异。任何生产落地前独立验收与来源迁移另报，不在本次热修。

## 6. 预算、验收和回执
不增加本轮正式学习预算：活动分支30000、累计30100及既有各阶段上限保持。不存在Q008丢失更新，不得新增工程重放名目。
允许既有课程与最终300例测试按原规则运行，但最终报告须分别列行为结果、记录语义附注、完整隔离未建立这一工程限制；不得只写“全SDS无条件通过”。行为指标可以真实通过或失败，O-15不会被另一个结果自动关闭。
回执本裁决SHA、仍运行的run/当前update、只读消费者审计负责人、sidecar路径及400解释；600完成后回送两个集合各速度成功数/failed/streak/stop，若原规则触发停止则照实报告。实施/分析子agent与独立D分离。未覆盖问题问主Codex；不commit/push，不恢复R2seed43。
主Codex本轮只读核查代码、400原始记录与D的RNG调查，没有运行新的GPU/仿真或生产测试，也没有修改活动源码。对行为无影响的判断限定于已核查的保持集分支，完整RNG隔离及训练最终效果尚未验证。
