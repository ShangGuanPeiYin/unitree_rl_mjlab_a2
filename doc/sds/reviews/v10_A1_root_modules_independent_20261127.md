# A1 根模块独立审核（2026-11-27）

审核人：`/root/learning_stack`。本次独立审核对象由 `/root` 实现；审核人没有修改这些实现或对应测试。审核对象为 `diagnostic_a1.py`、`evaluation_capacity.py`、`checkpoint.py` 的 A1 身份逻辑。盲教师 yaw-only 适配由审核人实现，**不在本报告的独立批准范围**，须由另一审核者批准。

## 当前结论

上述三个根模块的 CPU 工程逻辑通过源码复核。新 A1 资源关联证书实现仍需单独独立审核；本结论不批准执行物理测试，不代表 GPU 资源准入、教师能力或行为验收通过。

本报告绑定的实际文件 SHA256：

| 文件 | SHA256 |
| --- | --- |
| `src/tasks/velocity/v10/diagnostic_a1.py` | `9fb5c774251535893f6d33300209f1ab75238e68d474fb2a9d4f3143ca252d72` |
| `src/tasks/velocity/v10/evaluation_capacity.py` | `221d49491f1d470944bab67d729e62bed49bbc1b139bf377aae10a1c34409c32` |
| `src/tasks/velocity/v10/checkpoint.py` | `12620a96dd69db42ad4c5769519cb89d7c65367b99dda38e581dfcb99675c737` |

后续这些文件变化必须重新核对；最终全源码指纹由合并后的独立资源审核或根最终汇报另行绑定，不将并行开发时的全仓指纹冒充最终冻结值。

## 核对结果

1. **精确诊断清单与成本。** 清单仅包含普通平地10组，以及高度2m、G编码、宽1.2m/0.8m的平台各9组；每组原始A集索引0～3共4例，即28组、112例。案例沿用原始实际初态、病例SHA与max_steps，没有挑成功seed。逐组四例max_steps一致，最大并行控制样本之和严格为251288，包括padding的最坏上界。诊断使用4/4条件并显式标记diagnostic_only，不能替代正式18/20。
2. **一次性约束。** 首轮预审发现只拒绝同输出目录、换目录仍可重复。根已修复：在容量获取后、GPU初始化前，于规范化费用账本旁以固定协议名 `O_CREAT|O_EXCL` 创建持久claim，并fsync文件及父目录；不同输出目录不产生第二次机会。失败/中断保留claim，需新裁定，不自动重试或调增益。
3. **资源门禁前置。** `run_diagnostic` 必须提供 `resource_selection`。它在输出、claim和物理执行前核对passed、N=1024、冻结SDS、当前源码，并调用新的关联证书验证。最终资源关联实现的正确性另审，不能仅凭此调用链认定资源已通过。
4. **容量与费用分离。** `evaluation_capacity` 读取历史charged，只预留251288所需的阶段容量；不按上界提前扣费，实际控制步仍由Q1账本计费。正常释放仅记录未使用容量，历史executed/uncertain不退款。全局评估cap固定1685390840，诊断不能自行扩大。
5. **并发/硬崩溃关闭。** 同账本容量采用独立文件锁，竞争立即拒绝。持久active租约在PID不存在时也不会自动释放；未决Q1步骤同样阻止新阶段，需确认旧执行已结束后显式恢复。
6. **教师身份。** 检查点保存及验证均使用实际 `teacher_identity()` 完整映射，其中包含blind/stairs权重SHA以及blind_adapter配置SHA，并继续绑定SDS、recipe、源码、N和物理摘要。旧仅两权重身份不再兼容；没有隐式补齐适配器或旧证书回退。

## 实际验证与边界

审核人独立运行了纯CPU临时账本检查：假dead-PID的stale active租约拒绝，pending步骤拒绝；已预扣4条保留且executed=0，未退款。未创建GPU上下文或物理控制步。

定向pytest执行时，已完成的6项通过；根并行新增的 `test_diagnostic_rejects_unlinked_resource_receipt` 当时因其实现尚未落盘而失败（6 passed、1 failed，0.68s）。该测试失败如实保留为开发时TDD状态，不写成最终通过。根随后实现 `validate_diagnostic_resource`，审核人已读取最终前置门禁代码；最终绿测由根正在进行的定向/统一CPU回归提供，不重复启动整组验证。

本报告不包含新的物理诊断、资源探测、教师准入、DAgger/PPO训练或成功率改进证据。ETH原始训练指令范围尚无原cfg证据；normalizer非零std不能证明A1指令处于原训练范围。
