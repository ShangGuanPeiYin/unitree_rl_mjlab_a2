# Q008 after 完整工程证据独审（2026-11-20）

独立结论：适配 checkpoint、after DIAG1500、Q012 数值解除、production 四路 comparison 和 adapt75 全部录像的既定工程证据已齐全，核验通过。现有9d4批准及既定运行时条件继续适用，本报告不是新学习批准，不构成行为恢复、门禁通过、final资格或S1_RESTORED结论。未读取封存final成绩。

固定 C9源码 `c9ee8789f6051bc6c36cbfc24f076deba6121d8ffedb48ce5d689e0a6b665ea4`、SDS `500ad32aaa946c927459427f3a65565b15d4b77fca4be214817121d15c742976`、唯一6400 `30ed23ba16f7e490afca001fbbecadb4a11ff4b11733a5ae7ac7d0c841f9a8b2`、同Context `0d158…0408`。适配CP完整SHA `3453350b921e81d0e3e568a3d6c93a6044b8c6b62edc6d3362083a0a0235d800`。已有独立CP证明核实813轮、19,980,288控制样本、每轮20Adam，18项state各16,260步；18学习参数变化，35项对实际32c初始精确冻结，含std及normalizer。RNG/角色、journal和已写ledger前缀绑定、全部813轮有限性已验收；本次复核其固定证明SHA，没有重新GPU恢复或重跑训练。

after DIAG75批/1500唯一病例、75PT与75编译几何证书、首次终态、metrics、实际命令/action历史和teacher/student非地图同状态输入已在独立原件证明中核实。15110条保存审计状态不等于物理或学习样本。费用812240=首次回合723322+padding88918；现活账本仍恰75已结complete操作、义务closed、无重复计费。旧诊断子3350375实际已回收。保留原pending报告和CPU例外原文；Q012以新独立签字解除其数值等待，两份文件未回写。

production comparison 与独立CPU参考使用相同两个checkpoint、相同保存state/原教师标签及固定sV=2.136540383972894。动作项为12维MSE，值项为按sV归一后MSE，总训练目标为动作+0.1值。有效state逐条累加再按原任务/lesson/速度/停车组分母平均，全部四路各75组/1500病例分母一致、600项标量finite且非负。以下为正式GPU组结果按state数加权的汇总，完整75组同数据差值保存在新proof：

| 保存状态集 | 模型 | 状态分母 | 动作MSE | 归一值MSE | 动作+0.1值 |
| --- | --- | ---: | ---: | ---: | ---: |
| before | initial | 9366 | 0.0181318998081 | 0.0113713542278 | 0.0192690352309 |
| before | adapted | 9366 | 0.00452945559196 | 0.00189450787185 | 0.00471890637915 |
| after | initial | 15110 | 0.0710282139573 | 0.0271826687095 | 0.0737464808282 |
| after | adapted | 15110 | 0.00203751550761 | 0.00226102859421 | 0.00226361836703 |

同一状态集分别比较initial/adapted；双方自己的轨迹诊断为initial×before与adapted×after，其保存状态分母不同，不能将两个轨迹loss当成同分布效果证明。1500对行为按冻结case顺序及相同streams精确核对，canonical SHA `1f8e06aa7bd578083fd1220c05b839779669b09300cb0a0d207d7d1c63df1b26` 与预存独立参考一致。仅报告实际DIAG计数：before成功382/失败1113/超时5；after成功1051/失败449/超时0。这些不是SDS恢复门槛，不能替代固定gate/final。

按Q013，正式GPU值与CPU参考各自保留，不移用Q012逐输出tol或增加跨后端聚合loss门槛。600项最大GPU/CPU绝对差仍为8.06073769039628e-5：initial×before、stairs/L40/vx0.5的value，GPU0.7059099369992813、CPU0.7059905443761852。核验公式、原件、归约和分母未发现错误；未证明具体算子原因，未保存的四路中间GPU输出不能被本次CPU审查声称逐点重建。

adapt75全部冻结DIAG case0、同CP345335/同Context/C9/manifest闭合。75个MP4全部由纯CPU软件ffmpeg `-xerror`、1线程完整解码，合计18118帧；每件内容SHA、verification/metadata与完整frame_count一致。逐帧metadata核控制步、20ms时间、初始帧、每两控制步25fps采样及奇数终止补帧、准确pre-reset终态qpos摘要，flat Q002 i499 DECEL/i500 HOLD、stairs出生APPROACH及终态TOP_ARRIVED条件/恒定指令核对通过。640×480/25fps，75次N1工程操作已知complete，费用36049控制样本与summary/ledger一致，义务closed，无nonfinite或contact overflow工程掩码。未从像素反演完整物理姿态，也未声称从摘要重建未保存的中间qpos或楼梯空间位置；原渲染生命周期和阶段字段依据冻结源码及原件元数据审查。

新增证据本身未执行GPU、环境、物理、信号或optimizer；未修改源码/测试/既有报告/catalog/journal/ledger，没有提交。活账本核验使用共享锁同image，合法学习追加尾部不被当作旧证据篡改。观察快照seq5242，事件 `d8606306cc8506e3227af4fe851c539bce6df574a793a71deec8c4ade4890c46`；之后运行进度以外部账本为准，不由本报告回退。

新综合proof：`q008_after_complete_independent_proof_20261120.json` SHA `f83b1143e798acdefedfb3be1ebbab086940de4fe8ebabe60f1de34642004b7d`；视频proof：`q008_adapt75_videos_independent_proof.json` SHA `e672573a854aacb436d2a615597a300285c2350e53f6bd69d55a2c164d25a770`；comparison proof：`q008_after_production_comparison_independent_proof_20261120.json` SHA `8674abb4ec4fee8972a7de25914e60456c23a3fa793a02ba02eb77eea41802ad`。两套CPU/GPU数值原证据、Q012签字、Q013、CP、原件、录像和逐件完整解码receipt摘要均由proof绑定。共同日志与Git由root统一追加管理。
