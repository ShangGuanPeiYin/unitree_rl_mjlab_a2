# S1.2 入口与首个主段迭代独立状态审核

2026-11-20。结论：**指定 S1.2 入口及首个 main 迭代的数值、优化器和工程连续性核验通过，无阻断项。** 本报告不构成行为恢复、门禁/final 通过或 `S1_RESTORED` 结论。

审核绑定 `a2-v09-s1-seed42-r3`、C9 源码 `c9ee8789f6051bc6c36cbfc24f076deba6121d8ffedb48ce5d689e0a6b665ea4`、原 SDS/6400及同一 Q010 Context。仅执行纯 CPU 审核；没有 GPU、环境构造、物理推进、进程信号、训练暂停或重启。

## 固定 checkpoint

以下文件均位于 `logs/a2_v09/s1_seed42_r3/checkpoints/`，以同份已核 SHA 的 bytes 在 CPU 加载：

| 对象 | 文件 | SHA256 |
| --- | --- | --- |
| 实际初始 | `s1-q010-formal-20261120-r1_course_entry_000000000000_00000001_302e34dceb534f9187b9a6570e63114c.pt` | `32c962f28352b13f81c1c9b39e4c5a9bbe01f5e3f6545d98f8b46414492e6c13` |
| 适配完成 | `s1-q010-formal-20261120-r1_S1.1_adapted_000019980288_00001637_0b1ff92df94045cbbd4f6bc788b48768.pt` | `3453350b921e81d0e3e568a3d6c93a6044b8c6b62edc6d3362083a0a0235d800` |
| S1.2入口 | `s1-q010-formal-20261120-r1_S1.2_entry_000019980288_00001642_1115b45d9d80407fb3746470a1a89dc7.pt` | `ce7dc5a397183a300111679b40908a9079b6463ef42b2c441545968cc415f1e5` |
| 首个main迭代后 | `s1-q010-formal-20261120-r1_latest_000020004864_00001646_ac3575c92b1d48ad99358aa03ae9e13d.pt` | `4d542a676f61e9d1f880671b3992992988e545c741b2060e67ed4fb4cf913da0` |

## 实际验证

入口全部53项模型 state、完整 optimizer、角色计数以及 Python/NumPy/CPU Torch/CUDA编码/shuffle/environment RNG 与适配文件精确相等。18项既有 Adam state 均为16260步，入口没有执行额外更新或重建数值状态。入口模型 state SHA 为 `b2fbd7ed03e4bc32f8352562f8ada82a1f574a40b1aebd7dff55d85327ae926c`。

首个 main 迭代后，Adam仍管理同序45参数：旧18项各16280步，新26项各20步，std无Adam state；44项可训练模型参数均产生变化，模型及动量张量全部有限。四份文件的std和8个normalizer buffers共9项，逐张量与实际初始精确相等。

固定设置保持512环境、48步rollout、每轮24576样本、5 epoch×4逻辑minibatch＝20次Adam、microbatch512、LR＝3e-5、原Adam配置及冻结sV＝2.136540383972894。目标仍为动作MSE＋0.1×归一化值MSE，未改损失、速度、课程或终止口径。教师控制按独立episode stream在重置时选择，S1.1概率0.50、S1.2概率0.10；这不表示有限512回合必须恰含10%教师回合。周期checkpoint仍按累计500万样本跨越后的完整迭代保存，archive周期为1亿样本。

适配813轮＝19980288样本；首个main轮增加24576，总814轮＝20004864。814条训练audit前缀的样本、20次Adam、分组分母和统计有限性通过，首个main audit与checkpoint及journal引用一致。

## 账本与Q009顺序

使用已闭合固定前缀核验，后续合法训练追加不被当成身份变化，也不由本报告回退费用：

- 入口checkpoint绑定ledger sequence5219，首main文件绑定5226；对应学习分配分别为S1.1＝19980288，以及另加S1.2_main＝24576。
- `diag_after`在5067关闭，`video_adapt`在5218关闭；首main在5223预留、5224完整结算24576样本，绑定入口checkpoint，无重复计费。
- journal依次为工程接受1641、入口checkpoint1642、Q009 `evidence_session_finished` 1643、首main物理提交1644、20次Adam完成1645、周期checkpoint1646。
- Q009开始收据记录环境及缓冲已释放；其训练fingerprint与入口模型/optimizer数值相符。结束收据记录`optimizer_replayed=false`及新物理episodes。首main后的角色计数恰按102/102/154/154分配推进512个新ID，外部租约保持单调。

ledger至5226前缀为9088527字节，SHA `83a35a30f5844721586978f42e76626aa3c716ce0fd1b2b59984360f09c7407f`；journal至1646前缀为28209355字节，SHA `7d72bf04cb50ba0ef51c513d80e73b8635450f8ff8b808eea1b5ec2b8a541e11`。production comparison与adapt75完整性沿用已有独审证明，本次核其固定身份和闭合顺序，没有重复视频解码或数值重放。

## 证据边界

`state_dict`不保存活参数的`requires_grad`、Python对象ID或GPU恢复执行过程。CPU实例化相同C9模型确认S1.1为18项、S1.2为44项可训练参数；实际44项Adam状态及更新提供交叉佐证。Q009原model/Adam/ID allocator及参数映射、RNG恢复由未变源码中的运行时守护和已完成收据佐证，不冒称本次直接读取了活进程对象ID或重放了CUDA/environment RNG恢复。CPU RNG仅在审核进程本地验证可装载，CUDA编码只作CPU检查及精确比较。

本次结论仅覆盖上述固定checkpoint及前缀，不外推持续运行的后续全部训练。未读取封存final原文，没有新增学习批准或行为改善结论。

本机详细proof：`logs/a2_v09/s1_seed42_r3/s1_2_entry_first_main_independent_cpu_proof_20261120.json`，SHA256 **`dc6eb862f823759a6a6f6fed821df8407e582d0dd4759c7c6f426841ab29a4a0`**。报告依据该已完成审核整理，未重新执行测试；共同日志及Git由root统一管理。
