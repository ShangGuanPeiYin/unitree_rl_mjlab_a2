# A1最终工程聚合独立审核（2026-11-27）

## 最终结论

**最终冻结源码的CPU工程审核通过，无未解决阻断；可以由根按已批准的预算和顺序申请/签署精确绑定的工程与资源关联文件。** 本结论不跳过实际新资源短测、112例诊断、完整教师准入或行为门禁，不代表这些物理环节已经通过。

最终源码指纹：

`2f82589e805b2579843a4381089467d40931300aff8a80f466becf90dcc4fbcb`

根最终统一CPU结果已只读核对：**159 passed、5 skipped、1上游torch warning，87.45s**，原始日志为 `logs/a2_v10/A1_engineering_20261127/final_v10_cpu.log`。运行显式关闭GPU及full-replay opt-in，不能把这些跳过项写成新物理通过。

审核人 `/root/learning_stack` 聚合其他实现者的独立结论，并独立检查最后的晋级取消增量。本报告不将审核人自写的盲教师或旧公共资源core作自我独立批准：教师依据另一审核者的结论；旧core依据既有独立审核及固定SHA的旧完整b48双栈证据。

## 既有报告与历史保留

以下报告原样保留，不覆盖其中的历史状态：

| 报告 | SHA256 | 本次使用范围 |
| --- | --- | --- |
| `doc/sds/reviews/v10_a1_teacher_trace_engineering_review_20261127.md` | `8606fd834b76a099590250304ecbd26b5d94d70584984a91587a7ced8a4f2cb7` | 非教师作者对A1公式/私有clone/原ETH前向/混合路由的独立审核；trace作者自己的测试不冒充独立签字 |
| `doc/sds/reviews/v10_A1_root_modules_independent_20261127.md` | `71dc66acec54fc91af3b79a30520503ea40428eef51700ab47e3a806d3f3afec` | 根实现的112精确清单、一次性claim、容量/费用分离与checkpoint A1身份 |
| `doc/sds/reviews/v10_A1_resource_inheritance_independent_20261127.md` | `7bb97b5fb1335b4cb26e1845a58d9dc82581ea1bb969726d0da460ae9d31be73` | D90资源关联/授权承继与完整身份容量修复的独立工程审核；保留D90与156 passed的历史身份 |

D90证据归档于 `logs/a2_v10/A1_engineering_20261127/intermediate_d90`，8daf证据归档于 `intermediate_8daf`。审核人对比D90和最终295文件逐项SHA记录，**仅 `orchestration.py` 与 `test_v10_orchestration.py` 变化**，即下面审核的晋级取消增量；不把D90结论直接冒充最终2f源码结论。

## 最终源码与三项证据绑定

路径基于 `logs/a2_v10/A1_engineering_20261127/`。

| 证据 | SHA256 |
| --- | --- |
| `actual_baseline_reconstruction.json` | `2e25a585a2c4dee2d2953e95866e7500ca6b49e31cb6b4321e5c7949740e2dc4` |
| `resource_evidence/invariants.json` | `9bb100392a210d2c49b3a50a6a8b28cd6e5b6cfda3db64795e0d7c44e3788ee5` |
| `resource_evidence/source_diff.json` | `420aab6f9d7c3d47dc8c167fc0b756ff3a560890425df9953a1b1d1af709af93` |

审核人重新读取全部295文件，独立复算最终指纹严格等于2f；按证据将12个A1修改文件替回07fe基线blob、排除11个A1新文件、维持其余272个文件实际字节，严格重建旧实际指纹：

`b48e3fd1601ff2926272ed014d846fd480fe24a8c56d77100d646f933b92991f`

三项文件SHA均匹配上表。42项核心/资产不变量继续满足：41项实际字节相同；env字节不同如实保留，仅允许精确指定记录语句的严格AST例外，不泛化忽略任意frame操作。没有清理既有dirty/untracked或将其错误归因于A1。

## 最后晋级取消日志增量

独立阅读实际 `_promote_course`、事件应用/回放和生产调用点，并读取定向原始测试 `/tmp/v10_promotion_final_green.log`：**36 passed、1warning、37.98s**。确认：

1. 本增量作用于DAgger的D1/D2/D3课程晋级；只在当前checkpoint SHA与grade一致、A/B完整存在、teachers两套比较均通过且其candidate_model_sha一致后进入。
2. grade保存为不可变证据JSON，journal引用checkpoint和grade SHA。单个 `course-promotion:{course}:{target}` 事件同时写courses及可选promotion_cancellations，不能崩溃后只晋级而丢取消事实。
3. 只枚举当前course的后续target、A/B、candidate/teachers调用；operations中已经出现的调用均不取消。没有剩余target时仍可记录晋级，但不制造取消记录。
4. 取消记录明确原因、触发事件、checkpoint SHA、套SHA和原case最大控制样本上界；这是计划调用上界，不是实际退款或实际cache节约。charged/refunded/transferred均为0，不修改费用账本，不删除共享教师cache。
5. `courses` 与 `promotion_cancellations` 都按journal权威回放；重复事件和旧snapshot恢复幂等，checkpoint字节不变。测试覆盖started排除、末gate无取消、同一晋级重复及旧snapshot重放。真实比较函数提供所检查的candidate_model_sha字段，未仅依赖测试自造字段。

该增量不改奖励、损失、课程配方、速度、并行数或实际评估门槛。PPO提前结束路径保持原样，不把本次课程晋级日志扩大描述成新PPO算法行为。

## trace与工程门禁的聚合判断

根已对trace/env/eval_worker独立通读并要求将逐帧dict改为连续预分配数组；其定向及recording复核证据位于本工程目录 `a1_root_trace_review.log`、`a1_recording_review.log`。本审核人另只读核了真实 `TraceBatchV2` 与 `run_suite` 数据流：

- action-time请求/route/yaw与post-step yaw/接触明确分离，blind私有命令直接复用生产helper；candidate的teacher_command_valid=false。
- 每字段连续定长数组保留源dtype，shape/finite/mask/容量错误拒绝；保存只取已记录区间的view，没有完整轨迹concat。
- 首终止帧先入trace再冻结；后续padding不写病例trace，不重复结算；旧grading rows20不被新侧通道替换。

结合其他作者的教师独立审核、根trace复核及最终统一CPU回归，当前工程接线未见阻断。NPZ压缩与新增记录临时开销仍必须被真实资源监控覆盖。

## 执行边界

- A1资源仍是明确批准的“旧完整1+5+20证据 + 新1+2+2定向补证”关联，不是重新完成20次测量。N1024两栈各245760、共491520；任栈失败停止，无512回退或重复凑余量。
- 同cap承继授权子SHA为 `49760a42facac7017b88db514b3beb09f8c7fb9b75458ac932a3a7bcc86d2d6c`，父SHA为 `8afde6347fe88002da3a8bf542c080ff65f0ae52fbd47d8660cbd39fa67035e2`。原cap3833856、旧charged3293184不退，余额540672；本审核只读核查，没有begin/reserve生产序列。此前证据不足、旧权重/旧模型缓存不得因文件名相似自动沿用。
- 固定112例诊断上界251288，4/4只作单次诊断停止规则，不替代正式18/20；必须先过精确源码资源关联门禁。一次性claim与global protocol attempt持续阻止换目录/序列重试。
- 正式每阶段容量只扣完整同病例/模型/源码/物理身份且资源通过的完成receipt。旧T_blind模型身份、partial、缺例/重复例均不能抵扣；预留容量与实际Q1费用分开，不退款。
- 评估cap1685390840、旧304100不退。全程零缓存最坏1685946228、潜在缺口555388仍如实保留；阶段余额不足必须停，取消上界不等于额外可花额度。

**A1新GPU短测、112例真实诊断、完整教师准入、DAgger/PPO及效果改进仍未由本审核验证。** ETH原训练指令范围缺原cfg证据，normalizer非零std不能证明其支持范围。本次最终通过仅是CPU工程与执行防线审核通过；根按新source/本报告SHA生成最终JSON并完成提交推送后，仍须逐项实际通过物理门禁。
