# v0.9 Q010 S1 学习独立准入（2026-11-20）

结论：**批准当前固定 run、C9源码和N=512下，自动执行S1.1及满足运行时工程条件后的S1.2，包括既定Q008证据、门禁与资格受控终验流程。** 这是学习入口批准，不是S1行为恢复或S2晋级证明。当前正式学习为0；本独审没有启动物理或学习、没有读取真实final成绩。

批准范围为 `S1.1_learning`、`S1.2_learning`。批准对象是本文末尾固定命令及其既定串行子流程，不能用资源probe状态启动正式训练，不能自行改变配方、预算、阈值、源码、病例或兼容清单。实际S1.2仍由下述运行时条件阻断；工程异常或不可判结果须停报，不能用本批准跳过。

## 绑定证据与当前条件

- run_id：`a2-v09-s1-seed42-r3`；resolved run_dir：`/home/xxs/dog/unitree_rl_mjlab_a2/logs/a2_v09/s1_seed42_r3`。
- 当前源码：`c9ee8789f6051bc6c36cbfc24f076deba6121d8ffedb48ce5d689e0a6b665ea4`；独立代码审核ready，无未解决阻断；root完整CPU回归239 passed，既有Torch警告，源/测试前后不变。沿用已完成审核，不重复执行大套回归。
- SDS：`500ad32aaa946c927459427f3a65565b15d4b77fca4be214817121d15c742976`；6400：`30ed23ba16f7e490afca001fbbecadb4a11ff4b11733a5ae7ac7d0c841f9a8b2`。
- 同一只读兼容清单：`q010_s1_0_evidence_identity_c9ee.json`，SHA256 `0d158b266dde7daeefaf81bc5b270a4294a94f473203015ea8fd9d6df6380408`。旧S1.0基线、五套教师缓存和独立校准已结账封存；旧producer447不改，final只做索引/哈希检查。完整性及新旧源码/依赖连续性见此前固定独审报告。
- 实际资源报告：`resource_q010_r1/resource_admission.json`，文件SHA256 `5303ffdeaa8b3f22caf87a0a527c3f165400661d52b24ebbfe2fd93e532628b9`，语义SHA256 `d712183d48eaa89db4c76209cbf4e52b99d8bd83d873b724a3e409c599e17f3b`。容量独审报告SHA256 `2b8882c71194c85c6dc32e6c78ea4c5bb1801d224491da7b595dc25a73586f76`。512完整工作及Q009布局通过；1024没有容量资格，不得擅自改N。
- 校准保留原值 `s_V=2.136540383972894`，来源为独立600例的首次回合值，排除padding，未使用final确定训练参数。

本次新调用真实 `validate_admissions` 和 `validate_schedule_ledger`，只用CPU且不构造trainer。两者通过；正式journal不存在、学习消费0、无pending操作，diag_before/after、六模型视频、11次候选门禁和3套候选final义务均存在、open且零消费。活账本seq1157/event `7b75a39666ab73a08e683bd0fa1ce662dc9fd3b950195d9ed3b0e0bbb7e8c0a5`，检查前后字节相同。证明文件 `q010_learning_preconditions_independent_proof.json` SHA256 `a81ee1f6a99537326e9118f52d5f9b824b13679f7e4e761248e6e2f12aae2250`。只读hash检查final文件不授予其原文读取资格。

root真实train CLI无execute预检已exit0；独审核对其proof及stdout/stderr摘要、账本未变。proof SHA256 `54e5b1210d433df7fd76d0a883a32cda5351f2e15b55ed9fffb9897562cb6a32`。该预检本身不产生学习授权；本报告与随后绑定本文SHA的批准JSON共同构成此次授权。

## 正式初始化与训练合同

正式trainer无resume参数，从唯一6400重新迁移新学生，重新创建Adam；不继承旧教师训练状态，也不读取资源探测的学生权重/优化器/RNG。地图输入为原生3×40×30，Actor/Critic两个FC按规定6格到5格重叠加权迁移，其余参数和归一化状态逐项复制。资源实测中初始模型身份一致；Q008还须对正式初始checkpoint逐张量验证同一迁移。

训练环境不改变原名义动力学、原楼梯终止或速度集合，无新随机化和新登顶后走停目标。角色约20%平地行走、20%平地走停（两方式均衡）、30% H00–H15、30%其余6类原楼梯，各角色按既定整数分配；每回合三档速度均匀选择。教师/学生使用同一真实执行轨迹的本体、历史与特权量，分别采样自身旧/新地图，教师标签为同状态12维确定性动作均值和1维value。

S1纯蒸馏：`L_action=mean_j[(mu09_j-mu6400_j)^2]`，`L_value=((V09-V6400)/s_V)^2`，`L=L_action+0.1*L_value`，按逻辑批完整样本数归一化。没有PPO或奖励梯度；原奖励仅供诊断。本次未改变原规定损失、奖励或权重。

固定seed42、N512、rollout48，即24576控制样本/迭代；每迭代5epochs×4逻辑批=20次Adam，每逻辑批6144样本，microbatch最大512，完整累积后一次clip_grad_norm=1.0、一次Adam。Adam LR=3e-5、betas=(0.9,0.999)、epsilon=1e-8、weight_decay=0。std和Actor/Critic归一化统计全程冻结，饱和率保留诊断，异常停报而不自行更新统计。

| 学习段 | 原学习样本上限 | 最大完整迭代 | 可消费完整样本 |
| --- | ---: | ---: | ---: |
| S1.1适配 | 20000000 | 813 | 19980288 |
| S1.2主恢复 | 630000000 | 25634 | 629981184 |
| 终验行为失败后retry_1 | 100000000 | 4069 | 99999744 |
| 再次终验行为失败后retry_2 | 100000000 | 4069 | 99999744 |

S1.1仅地图编码器和Actor融合分支18参数学习、50%教师回合；S1.2解冻Actor均值网络和Critic全部44参数、教师回合降至10%，沿用同Adam及已有地图动量，std继续冻结。课程切换使用新物理回合和单调episode_id，不携带旧50%控制分配回合。达到行为条件可以提前结束S1.2；表内数为最大整数完整rollout，不是必须花完或已完成数量，未用预算不得挪用。

每5000000学习控制样本跨阈值后的首个完整rollout保存checkpoint、保留最新5份；每100000000学习样本归档，课程/资格边界另存并去重。普通保存不重置回合。外部账本先预留后结算，部分/未知rollout保留费用且不进入优化；恢复不得回退累计费用、名额或失败计数。

## Q008/Q009是运行时硬条件

首个学习rollout之前，自动driver执行独立S1_DIAG全部75组×20例=1500例初始诊断，并保留6400/迁移初始各75个固定case0视频；这不是已存在结果。S1.1完成813个完整rollout后再做同一DIAG1500例和适配后75视频。两个模型交叉在同一批已保存真实物理观测及6400标签上计算loss，另报各自轨迹loss和逐例行为配对。诊断不选模、不占11次门禁，不要求此时行为已恢复。

只有最大完整适配量完成、loss/梯度/动作/状态有限、初始迁移逐张量正确、编译几何/接触索引证书与当前源码一致且无overflow/nonfinite、前后对照完整时，自动driver才可进入S1.2。任何证据缺失、失败视频丢失或工程异常均停报；不得把报告中的布尔标志代替原始证据。

外部诊断/比较/视频/门禁/final均在48步和20次Adam完整提交后执行：保存当前模型、同Adam、冻结统计、训练RNG和角色计数，记录已付费回合截断而不虚构成功失败；释放env与双缓冲并卸载模型/Adam后串行启动子进程。回来验证checkpoint/指纹/同Context、恢复设备和RNG，使用新单调ID回合。实际容量探测已覆盖这一布局；不允许训练env仍驻留时另启外部评估。

## 行为门禁与封存final

最多11次完整75组门禁，A/B按实际attempt交替，正式每组20例；主段计划阈值为1…6亿及6.3亿末尾，两个已解锁retry段各5000万和1亿。阈值映射到可达完整rollout，课程末尾取整下界；首过后冻结同一checkpoint SHA并立即用下一有效名额确认另一套，确认之间禁止训练。依Q006/Q007，可以消费未来确认名额及取消对应未来阈值，但不因此解锁retry学习；已结束段剩余阈值永久取消，恢复不复活。

两个相邻、工程有效、同SHA A/B均通过，才可解封当前下一套final，每组100例共7500例。每次实际读取final manifest或原teacher summary/episodes，仍须fresh活journal/ledger执行真实 `validate_final_unseal`，绑定当前候选和套别；本次批准没有提前解封final。候选视频仍使用独立DIAG固定病例，不能通过final选模。只有当前7500例正式终验行为通过才生成 `S1_RESTORED`；这不自动启动S2。

验收保持SDS3.5和Q001–004：全N成功数不低于6400；每楼梯组净前进P10/P50/P90及配对均值分别满足原5%/0.05m容差、每类物理失败不增加；平地速度RMSE容差max(5%,0.02m/s)、pitch RMS增加≤1°、步幅/步频中位数偏差≤10%、滑移/动作变化增加≤10%（零基线绝对容差1e-8）、停车漂移增加≤0.02m。每个指标按教师完整有效窗口集合T逐例配对，固定分母20/100；T为空不可判、候选缺窗/步态覆盖不足失败，不填0或删病例。质量判据、全巡航stall与周期汇集使用已冻结裁定。

仅正式终验的**工程有效且可判定的行为失败**消耗一套并解锁下一段1亿专用retry；最多两段、三套final。工程无效、Q001不可判、名额不足、cap耗尽、预算不足或恢复一致性异常均停报，不能自动解锁retry、加RL、扩视野、放宽门槛、加病例套数或借用别段预算。工程重跑必须遵守整套重跑及剩余非学习预算规则，不能免费执行。

## 预算、授权与未验证边界

S1学习cap8.5亿、非学习cap1.5亿分别约束；非学习冻结计划上界124142190，已发生资源5631298含unknown上界172032，账本累计计费21361950，正式学习0。未来义务预留保护继续有效，不能为重跑挪用尚需门禁/final/诊断；所有padding和失败实际步骤照实结算。上述工程及预检通过不证明蒸馏loss会改善或行为能在预算内恢复。

拟签本机批准JSON：`logs/a2_v09/s1_seed42_r3/s1_learning_approval_q010_c9ee.json`。先固定本文，再把本文SHA写入JSON；JSON不进源码快照，本文不引用其内容SHA，避免自引用。签发后以真实 `validate_learning_approval` 和同一Context再核对，验证结果另记本机收据。root负责统一修改日志、提交推送并核远端后执行；独审不启动命令。

固定允许命令（首次fresh初始化，不带resume）：

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
CUDA_VISIBLE_DEVICES=0 /home/xxs/anaconda3/envs/a2/bin/python scripts/train_a2_v09_s1.py --run-dir /home/xxs/dog/unitree_rl_mjlab_a2/logs/a2_v09/s1_seed42_r3 --run-id a2-v09-s1-seed42-r3 --segment-id s1-q010-formal-20261120-r1 --resource-admission /home/xxs/dog/unitree_rl_mjlab_a2/logs/a2_v09/s1_seed42_r3/resource_q010_r1/resource_admission.json --baseline-admission /home/xxs/dog/unitree_rl_mjlab_a2/logs/a2_v09/s1_seed42_r3/baseline_admission.json --calibration /home/xxs/dog/unitree_rl_mjlab_a2/logs/a2_v09/s1_seed42_r3/calibration/calibration/calibration.json --code-sha256 c9ee8789f6051bc6c36cbfc24f076deba6121d8ffedb48ce5d689e0a6b665ea4 --evidence-identity /home/xxs/dog/unitree_rl_mjlab_a2/logs/a2_v09/s1_seed42_r3/q010_s1_0_evidence_identity_c9ee.json --evidence-identity-sha256 0d158b266dde7daeefaf81bc5b270a4294a94f473203015ea8fd9d6df6380408 --num-envs 512 --device cuda:0 --seed 42 --rollout-steps 48 --epochs 5 --minibatches 4 --microbatch 512 --learning-rate 0.00003 --checkpoint-every-samples 5000000 --approval-report /home/xxs/dog/unitree_rl_mjlab_a2/logs/a2_v09/s1_seed42_r3/s1_learning_approval_q010_c9ee.json --execute
```

本批准限上述S1及其受控自动子流程。所有旧原文、既有兼容/容量/工程准入报告保持不变。未验证事项为正式S1.1/S1.2学习结果、实际Q008前后行为和正式门禁/终验表现；本报告不声称效果改善或恢复成功。
