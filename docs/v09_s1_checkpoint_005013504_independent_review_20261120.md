# S1首周期checkpoint独立工程审查（2026-11-20）

结论：**指定5013504学习控制样本checkpoint的工程状态与冻结合同一致，审查通过，无新增阻断。** 本次只用CPU读取实际文件、实际初始checkpoint、权威journal及共享锁捕获的账本image；不启动GPU、环境或物理，不暂停运行父进程，不修改生产源、现有签署报告/catalog、账本、共同日志或Git index，不提交。

本结论验证工程一致性，**不是行为恢复、S1.2资格或S1_RESTORED证明**；没有运行评估、解封final或实际恢复GPU随机状态。

## 对象与身份

- 文件：`checkpoints/s1-q010-formal-20261120-r1_latest_000005013504_00000415_d5ef2172b7884717b2c0c73bf5cac924.pt`。
- 实际文件SHA256：`e43257834144000f5f8b810fcd7539c1b9d04333bee4098226f9d95d6ab327a1`。先对原始bytes核摘要，再从同一bytes按CPU加载schema11。
- run：`a2-v09-s1-seed42-r3`；code：`c9ee8789f6051bc6c36cbfc24f076deba6121d8ffedb48ce5d689e0a6b665ea4`；SDS：`500ad32aaa946c927459427f3a65565b15d4b77fca4be214817121d15c742976`；6400：`30ed23ba16f7e490afca001fbbecadb4a11ff4b11733a5ae7ac7d0c841f9a8b2`。
- 同一Context清单SHA256：`0d158b266dde7daeefaf81bc5b270a4294a94f473203015ea8fd9d6df6380408`。实际调用Context工厂及formal checkpoint identity校验通过，baseline/calibration Ref与该清单严格相同；不重复解析大校准JSON，不读取final原文。
- header的recipe、architecture、原生`3×40×30`及code_version正确；metadata为S1/S1.1、N512、s_V原值`2.136540383972894`、`resume_mode=new_physical_episodes`，不是probe状态。

204次完整迭代×512环境×48步=5013504学习样本；203次为4988928，小于5000000，因此第204次是首个跨过500万保存阈值的完整rollout。文件名中的`00000415`是journal保存事件序号，**不是checkpoint数量**。

## 模型冻结与Adam

以实际course_entry文件为对照，其SHA256为 `32c962f28352b13f81c1c9b39e4c5a9bbe01f5e3f6545d98f8b46414492e6c13`。全部53项模型state具有相同key/shape/dtype且有限；依据当前C9的S1.1合同重建18个可训练参数张量，实际变化仅发生在这18项。其余35项逐项`torch.equal`精确相等，包括std和8项Actor/Critic归一化buffer。迁移provenance与实际初始checkpoint完全相同，仍记录fresh6400迁移、未加载旧训练状态；该provenance描述初始化，不把其中初始model SHA误作已学习模型SHA。

| 检查 | 实际结果 |
| --- | --- |
| 当前可训练参数张量集合 | 地图编码器/Actor融合分支，共18 |
| 冻结state项 | 35，其中std+normalizer为9项 |
| Adam参数组 | 1组、45个唯一参数，包含冻结参数 |
| 已分配Adam state | 恰18项，与可训练集合相同 |
| 每个Adam state的step | 4080=204×20 |
| 其余参数 | 无Adam state；std亦无state |
| exp_avg/exp_avg_sq | shape/dtype匹配参数，全部有限，平方矩非负 |

Adam LR=3e-5、betas=(0.9,0.999)、epsilon=1e-8、weight_decay=0，未改变。std/归一化不更新的设计保证本次仅适配地图；优化器预先管理全部45参数，供后续合格切课时保留地图动量。

`requires_grad`本身不由state_dict序列化；这里的18项集合由审定C9代码、S1.1元数据、实际Adam分配和冻结值交叉核对，不声称直接读取了运行进程的梯度标志。文件不含每步梯度，独审没有重演204次反向传播。

## RNG、角色及写前账本绑定

Python、NumPy及Torch CPU RNG已在独立局部对象中装载并逐项roundtrip；Torch CPU状态5056字节。单个可见CUDA设备状态、独立shuffle状态、environment状态均为完整16字节seed/offset编码，seed42，offset分别4、25071600、0。shuffle相对实际初始状态已推进；GPU状态仅核格式和一致性，**未在GPU恢复或重演**。

environment offset0是合法未消费状态：实际V09按role/单调episode_id/named stream生成完整病例，将出生/速度写入initial_cases/initial_command；旧V07随机出生分支因此不执行。不能把这个零offset当作训练未发生或RNG丢失。审计器曾额外假设其必须>0，核源码后只纠正/tmp审核器的无合同依据假设，生产数据、阈值和配置未改。

角色配置精确为flat_walk102、flat_stop102、H00_H15共154、其余6旧楼梯共154。保存的next/claimed_end依次为1020/1024、741/1024、5019/5120、2982/3072，均为合法整数、next不超过已预领边界；各claimed_end精确等于保存账本前缀的对应角色计数，当前外部边界不小于保存值。未用ID租约不能靠载入旧checkpoint回退或重放；正式恢复使用新物理回合并以外部账本为准。

checkpoint绑定的账本已写前缀为seq2443/event `d48c4e77175ecaeef99b4ae960114e22c713ba4430127ae016ea862b25cc62a9`，前缀7737998字节，SHA256 `18d57dfb84ff748862589b3848c6dd5e684d52655d9198573fae7ee7bb40a372`。完整metadata.stages与从该前缀重算的状态精确一致；S1.1学习5013504、204项全部完整结算，其他学习段消费0。前缀累计**计费**26972753，不能因既有资源unknown上界而把全额称作精确物理量。

一次共享锁捕获的当前image已合法追加至seq2932、学习7962624；较新尾部是正常增长，未要求活文件仍等于旧前缀或从checkpoint回退外部消费。journal seq415保存本文件，schedule与metadata相同；之前204次optimizer_complete各20Adam，最近一条为414。所存最近loss报告摘要与该journal引用一致，loss/梯度及按任务速度分组统计有限，组样本分母均24576。以上是状态/记账一致性，不作loss改善或行为效果比较。

## 证据与范围

新证明：`logs/a2_v09/s1_seed42_r3/checkpoint_005013504_independent_proof_20261120.json`，SHA256 `4bd47530f7df414581f441e6a1b28244aab9eec987270e1559954bb8fa8bac97`。其中保留可训练/变化/冻结项、Adam检查、RNG编码、保存前缀和当前快照身份，未保存或改写权重文件。

所有检查均CPU完成，现有Torch FutureWarning不影响本次验证。运行仍在S1.1；后续完整813次适配及Q008 after诊断/视频/同数据比较仍按既有批准执行，S1.2、正式门禁和final资格未由本报告替代。日志与提交由root统一处理。
