# Q012：适配后数值核验待裁定（2026-11-20）

状态：S1.1适配已完成，适配checkpoint工程核验通过；after首批CPU教师标签复算出现原容差例外，整套after数值验收尚未完成，S1.2保持阻断。本文记录实际事实与待裁定问题，不是规划裁定或新的学习批准。

## 实际适配状态

固定C9源码、r3、源6400、SDS与同一Context不变。原N512、rollout48、每轮5epoch×4逻辑minibatch（20Adam）、micro512、LR3e-5、std与normalizer冻结，均未调整。S1.1完成813个完整rollout、19,980,288学习控制样本，未超20,000,000上限；每5,000,000学习样本在完整迭代边界保存，已产生5,013,504、10,002,432、15,015,936三个周期文件。

适配文件：`checkpoints/s1-q010-formal-20261120-r1_S1.1_adapted_000019980288_00001637_0b1ff92df94045cbbd4f6bc788b48768.pt`，SHA256 `3453350b921e81d0e3e568a3d6c93a6044b8c6b62edc6d3362083a0a0235d800`。独立CPU proof `checkpoint_0019980288_independent_proof_20261120.json`，SHA256 `3401b0d5502db9674530d24f8e1609782931dd69cd308c33d6fe376da6abf74a`。实际53项state中18项学习张量变化，其余35项（含std及8个normalizer buffers）与正式初始32c checkpoint精确相等；Adam管理45参数，恰18状态各16,260步=813×20，模型及动量有限。全部813轮audit、对应journal摘要与完整24,576分母及损失/梯度/分组统计有限性通过；CPU RNG局部装载、CUDA编码及角色租约与写前账本绑定通过，没有额外运行GPU恢复或物理重放。

保存前缀seq4912、8,801,738字节，SHA256 `b11f1b421595c2f182d969cf44088dc6c6495e69da00fc3722a06799bc42c41c`，813项学习操作全部known complete；metadata阶段与累计费用对齐。上述仅工程状态，不表示行为恢复或S1.2资格。

## after首批实际例外

固定DIAG的after batch0已结算20例；纯CPU独审重算其全部raw metrics、几何证书、有限性与同状态非地图输入通过。20个保存快照、每个20槽，共400个教师状态。使用原比较条件：

`abs(recomputed - saved) <= 1e-5 + 1e-4 * abs(saved)`。

动作162/4800分量超差，最大绝对差`6.966292858123779e-5`；value 0/400超差，最大绝对差`1.8596649169921875e-5`（其相对项仍使原条件通过）。首个异常为index100/full20，5个动作分量超差，动作最大差`3.975629806518555e-5`、RMS`1.0446883607073687e-5`。不将绝对误差单独当成含相对项容差的失败判据。

本机proof `q008_after_first_batch_cpu_numeric_20261120.json` SHA256 `9aecff60ad4c067edf2468efd4d15b0170949efa57b5438a548a12fb495e584b`，对应PT SHA256 `617fee5963930a29386893542b4f49e69cf7038574a363f2a50a9add87a23988`。未放宽原tol、改标签/原件或复用before Q011作为新数据豁免；当前不能确定具体后端算子根因，也未执行新的GPU重放。

## 安全边界与通信阻断

原父PID3228141/start_ticks145922371、完整argv与launch相同。journal1640为after evidence_session_started，env_released/buffers_released均true，complete_rollouts813、main学习0；父在do_wait等待唯一diag_after子3350375，子绑定上述适配SHA。root先写入并fsync hold intent，核同tick/argv/等待点后只对父发一次SIGSTOP，确认父T；子start_ticks147088853继续R/S，未对其发送信号。append-only `q012_parent_hold_20261120.jsonl` SHA256 `54361dbab6e232b075611705c8afe9d7826cef1f85ce9f1819a1d9a2526479b3`。没有重启、权重修改、费用回退或虚构回合结束。

Q012拟请求：允许after DIAG自然闭合，冻结本次完整CPU异常清单及数值合同，保留原tol/生产配置；参考Q011实施另行批准的零物理、同GPU、原N20布局双进程核验，完整/部分批及遮盖控制范围由规划明确；实际独审签字后，仅恢复原父一次，继续原9d4批准。此处为待决方案，未获裁定，禁止据本文运行GPU或恢复父。

向指定规划会话`01a1f385-ceea-74a0-a77c-85f66b2ccd77`发送该请求两次被自动审核拒绝，消息均未发出。首次认为当前用户仅授权项目修改/push，未授权该目的地的内部训练数据通信；只读核对当前对话及规划原始用户记录，找到2026-11-20T02:56:12.247Z用户明确要求执行主遇SDS疑问回问规划的原文。附该依据重试仍被拒绝，理由为历史工具输出不能满足当前通信授权验证。此后未重试、未改用其他渠道或agent代发，已向当前用户异步请求该目的地发送许可，等待答复。

不受影响的after DIAG自然采集及CPU只读完整性/差异枚举继续。整套after1500例、同数据双模型比较、75适配视频尚未验收，未放行S1.2、正式A/B或final，更没有S1_RESTORED、S2/S3或行为效果改善结论。GPU核验方案和实际恢复均未验证。
