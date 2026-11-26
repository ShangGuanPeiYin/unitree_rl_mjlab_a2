# A2 v1.0 实施裁定

关联SDS SHA256：eb95905ed24bf0674f94a1b02dab02256ee11e40a20ed1cd96eb24daeb1b49a0。
裁定来源：规划线程01a217c8-d825-7922-bd23-eb6ba2e37bfb。

## Q1：硬崩溃时的step费用（2026-11-26，已批准A）

每step先持久化唯一(run/attempt/rollout/step)预留N，成功后原子标为executed。硬崩溃未决归uncertain；charged=executed+uncertain限制课程及总硬上限，不退款、不重复计费。只允许一个step在途，每次崩溃新增uncertain不超过N。uncertain不能声称实际执行或训练成果，不进梯度/GAE。执行后未落成功标记也归uncertain，恢复不重演同ID。剩余不足完整rollout停止。门禁/保存阈值按已确认执行学习样本，在完整bootstrap/GAE/update后触发；charged耗尽但阈值未达，不虚构晋级，报告预算耗尽。须补预留后崩溃、执行后标记前崩溃、重复恢复幂等、预算边界测试。

## Q2：评估并行与padding费用（2026-11-26，已批准）

允许评估每组20/100环境、独立进程；统一1024/512仅约束正式学习及其双栈probe。同模型/同组/同max_steps组批，每次物理step全批B计费，含已终止lane的padding；最后病例首次终止时结束。首次终止立即冻结该例结果/轨迹/窗口，后续padding不重复结算或污染统计，关闭自动reset，padding动作固定零。上界显式为sum_batches(B*max_case_steps)，启动前断言组内max_steps一致且等于sum_cases(max_steps)；不等则拆同上限批次，不能静默扩预算。分别报告有效病例步、padding步，总执行/charged按全批实耗，Q1同样适用，不计学习额度。GPU峰值单独验证，训练不得并驻。补早停冻结、padding计费/隔离、上界等式测试。

## Q3：验收状态与已验checkpoint身份（2026-11-26，已批准）

候选checkpoint保持原字节，A/B及终验绑定文件SHA，部署交付原文件+sidecar；extra仅代表保存时状态。外部controller_state为原子快照，验收事件另追加持久化。证书绑定checkpoint、SDS/裁定、suite、物理协议、结果SHA。先持久化结果及验收事件/证书，再更新controller；恢复按唯一事件ID幂等重放并校验引用SHA，拒绝跨候选/跨suite合并。下一checkpoint嵌最新状态但得到新SHA，不能继承已验文件身份。预算、终验已用、回退次数以不可倒退外账为准，旧extra不能覆盖。补原文件SHA不变、错配/篡改拒绝、崩溃恢复及幂等测试。不改变训练配方。
