# A2 v1.0 实施裁定

关联SDS SHA256：eb95905ed24bf0674f94a1b02dab02256ee11e40a20ed1cd96eb24daeb1b49a0。
裁定来源：规划线程01a217c8-d825-7922-bd23-eb6ba2e37bfb。

## Q1：硬崩溃时的step费用（2026-11-26，已批准A）

每step先持久化唯一(run/attempt/rollout/step)预留N，成功后原子标为executed。硬崩溃未决归uncertain；charged=executed+uncertain限制课程及总硬上限，不退款、不重复计费。只允许一个step在途，每次崩溃新增uncertain不超过N。uncertain不能声称实际执行或训练成果，不进梯度/GAE。执行后未落成功标记也归uncertain，恢复不重演同ID。剩余不足完整rollout停止。门禁/保存阈值按已确认执行学习样本，在完整bootstrap/GAE/update后触发；charged耗尽但阈值未达，不虚构晋级，报告预算耗尽。须补预留后崩溃、执行后标记前崩溃、重复恢复幂等、预算边界测试。

## Q2：评估并行与padding费用（2026-11-26，已批准）

允许评估每组20/100环境、独立进程；统一1024/512仅约束正式学习及其双栈probe。同模型/同组/同max_steps组批，每次物理step全批B计费，含已终止lane的padding；最后病例首次终止时结束。首次终止立即冻结该例结果/轨迹/窗口，后续padding不重复结算或污染统计，关闭自动reset，padding动作固定零。上界显式为sum_batches(B*max_case_steps)，启动前断言组内max_steps一致且等于sum_cases(max_steps)；不等则拆同上限批次，不能静默扩预算。分别报告有效病例步、padding步，总执行/charged按全批实耗，Q1同样适用，不计学习额度。GPU峰值单独验证，训练不得并驻。补早停冻结、padding计费/隔离、上界等式测试。

## Q3：验收状态与已验checkpoint身份（2026-11-26，已批准）

候选checkpoint保持原字节，A/B及终验绑定文件SHA，部署交付原文件+sidecar；extra仅代表保存时状态。外部controller_state为原子快照，验收事件另追加持久化。证书绑定checkpoint、SDS/裁定、suite、物理协议、结果SHA。先持久化结果及验收事件/证书，再更新controller；恢复按唯一事件ID幂等重放并校验引用SHA，拒绝跨候选/跨suite合并。下一checkpoint嵌最新状态但得到新SHA，不能继承已验文件身份。预算、终验已用、回退次数以不可倒退外账为准，旧extra不能覆盖。补原文件SHA不变、错配/篡改拒绝、崩溃恢复及幂等测试。不改变训练配方。

## Q4：资源失败后的生命周期与保存路径修复（2026-11-26，已批准）

实际首次资源探测两档均因主机available<4GiB未通过：1024 DAgger executed98304，512 DAgger executed638976，合计737280；updated合计663552，uncertain0。512完成5预热+20测量及保存/实际恢复，但最低available2883108864。没有PPO probe、教师准入或正式学习。旧记录不得清除或返还。

规划批准只读定位及已证实重复驻留的内存生命周期修复：checkpoint和orchestration使用8MiB分块流式SHA，digest必须等价；完整保存并校验磁盘源后，释放旧replay/无用rollout/序列化缓冲再实际加载。真实恢复同样采用唯一所有权，不能仅probe特殊绕过；移除不必要deepcopy但不能产生可变别名污染。完整196608条CPU FP32、模型/Adam/归一化/RNG/计数及下一批采样必须一致，失败退出并保留磁盘源。采样/更新/保存/释放/加载分别记录RSS和MemAvailable，整个峰值统计不清零。禁止改变六×32768容量、48步、比例、模型、优化器或4GiB阈值。

诊断更正：实际运行4b80853的checkpoint.sha256已经是1MiB流式，不能将本次probe峰值归因于SHA整读；orchestration.file_sha整读是独立潜在开销。probe恢复保留旧replay同时torch.load新replay的代码事实与最低点位于最后更新后5.72秒的时间证据一致；旧日志缺内部阶段RSS，不冒充精确阶段归因。

另批准修复明确工程阻断：worker保存条件遗留未定义last_save_bucket，与已初始化last_saved_confirmed不一致。原工程通过结论追加标记发现遗漏、待重验，保留原报告。必须真实CPU worker集成覆盖首次跨阈值、恢复跨下一阈值、周期/阶段末重合去重、真实checkpoint落盘及SHA读取，只修本缺陷关联路径，不改变2457600保存周期。

独立重审后提交推送。重probe前核对全局工程预算；原execution_plan未明确全局B_probe，3833856只是按已有26*N*48单候选上限推导的保守上界，不表述为原文冻结值。规划正在向用户请求异常重跑额度，尚未批准时不得重probe；既有737280迁移全局持久账本且幂等不可退，学习/评估额度不能转借。最终probe总上限等待明确预算授权，不由实现者填写猜测值。

Q4补充（规划明确批准）：collector采用同形预分配CPU缓冲逐帧copy，replay按照原随机索引只取需要的256条，避免48快照+concat及复制整任务再抽样的临时副本。必须保持采集顺序、time/env展平顺序、dtype/device、同帧标签、当前/历史比例和RNG消耗顺序；buffer复用及env.step/reset不能覆盖历史快照，异步copy读取前完成。固定轨迹/随机状态的旧新逐元素/下一批对照覆盖跨episode、rollout末、reservoir更新前后和重复/有放回索引。全容量CPU测试注意主机可用内存，不干扰其他进程；无法完成须明确未验证。

预算状态补充：新增完整探测序列额度尚未得到用户明确答复。全局账本持久迁移历史737280，不产生新物理费用；授权文件缺失必须在所有probe入口第一步之前拒绝。3833856仅单完整序列的最坏工作量需求；拟议累计4571136尚不是获准上限，实现不得擅自写授权。学习196608000及评估1685390840上界均不变，不转借。
