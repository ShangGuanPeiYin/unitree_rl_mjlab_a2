# A2 v1.0 最终独立工程验收（2026-11-27）

## 结论、对象与证据边界

**工程代码与实际接线验收通过（包括本次发现后已修复的两项缺口）；整个资源/教师/策略行为验收没有通过。** 本报告不是正式行为证书，也不构成创建或绕过生产预算授权的许可。

- 基准已推送提交：`25575edfb47168fa45c1367673fd85f0f64a3962`。本次读验收时HEAD仍为该基准；发现后由主对话处理的定向修复已冻结但尚待其提交推送，不能冒称原25575ed已包含修复。
- 本次验收准确绑定冻结工作树源码指纹：`b48e3fd1601ff2926272ed014d846fd480fe24a8c56d77100d646f933b92991f`，不是仅依赖提交名。
- 基准：完整SDS v1.1，SHA `eb95905ed24bf0674f94a1b02dab02256ee11e40a20ed1cd96eb24daeb1b49a0`；Q1–Q4及文末2026-11-27 A额度/遥测补充裁定。
- 审核者：独立子agent `/root/review_preflight`，本轮未实现代码、未改配方、未运行GPU、物理probe或训练，只新增本报告。原独立审核及Q4追加报告完整保留，SHA仍为 `a82a34377bd3825fd2168bbe4dfbddf4847aabba137b2ad06a626827a9c97166`。

审核不是复述旧报告：重新从正式CLI追到controller、worker、collector、真实env.step、reward及终验/证书/恢复出口，并逐字节独立比较19个未改模块与基准提交；实际复核4个既有dirty保护文件SHA、两个教师文件SHA和SDS原文。它们均匹配。生产模块实际变更仅runtime/resources/probe_budget三项，对应本次两项缺口和必要预算lease修复；详见 `logs/a2_v10/final_acceptance_identity_20261127.json`。报告涉及行号均来自本次最终工作树。

## 本次查出的缺口及处理

1. **A授权数字与实际入口不相容。** 基准要求一次性为整个3833856最坏序列留足预算；若累计cap=3833856且保留历史737280，入口会因余3096576不足而在第一栈前拒绝，虽然能支付1024两栈。已按规划追加裁定改为每个26×N×48完整栈启动前独占容量；parent和child共享同scope lease。容量不重复计作charged，真正step仍按Q1逐条计费。不足下一完整栈立即停报，而不启动后再赌能够跑完。
2. **SDS§8.1训练遥测没有持久化出口。** 基准collector已有episodes及逐行metadata，但worker只写全体完成/失败总数后释放，无法重建每任务长度/失败比例/标签覆盖。已补完整collector→worker持久日志：真实完成回合steps统计、完整完成分母、未完回合gauge、teacher分项标签分母/覆盖、rollout/session ID；坏标签先作为工程错误停止，不缩分母；物理恢复的新回合不误记旧未完回合失败。这是日志与校验补齐，未改变Actor输入、loss、reward或课程。

本轮没有通过删旧代码、降低资源阈值、放宽成功标准或改教师来源使验收变绿。

## 固定关键配置核对

| 项目 | 当前生产设置 |
|---|---|
| 任务与速度 | F_WALK/F_STAND/F_STOP/P_WALK/P_STOP/S_ROUTE；比例20/10/15/15/10/30%；正速度0.3/0.5/0.6m/s |
| 观测/动作/控制 | 45维本体、5×45历史、3×40×30地图、3维路线；12动作；0.02s控制、4×0.005s物理、地图每5步刷新；q_target=q0+0.25a；原执行clip=None |
| 学习上限 | D1=24576000，D2=49152000，D3=73728000，P0+P1=49152000；合计196608000，不转借提前晋级余额 |
| 1024/512最大轮数 | D1=500/1000，D2=1000/2000，D3=1500/3000；P0=100/200；PPO总1000/2000（P1余900/1800） |
| 优化 | rollout48；DAgger每轮20/10次Adam；PPO每轮5×4=20次Adam；微批≤256；固定LR3e-5；梯度clip1；不把两档PPO Adam次数说成严格等价 |
| PPO | gamma0.99、lambda0.95、policy/value clip0.2、value系数1、entropy0.003、keep系数1；std初始0.10、投影[0.05,0.30]；不按KL调LR |
| 保存与门禁 | 每2457600确认学习样本越阈值完整更新后保存（1024每50轮/512每100轮）；DAgger40/80/100%，PPO25/50/75/100%；同SHA A通过后立即B |
| 准入/终验 | 每组门禁20例至少18；终验100例至少95；相对成功数/连续指标/配对覆盖率同时满足；最多两次PPO回退；独立终验不自动换集重试 |
| 费用 | 学习196608000与评估1685390840上界分账，评估71个suite/model计划调用；probe另有全局账本，不借学习/评估额度 |

## SDS → 实际生产接线 → 证据映射

“通过”均指源码/工程合同及相应CPU验证；没有把代码分支覆盖当作机器人行为成功。下表同时列出尚未实测的运行层面。

| SDS条款/工程要求 | 实际生产调用点及逻辑 | 测试或核验依据 | 结论 |
|---|---|---|---|
| §1.3、§13 正式入口及顺序 | [正式CLI](/home/xxs/dog/unitree_rl_mjlab_a2/scripts/train_a2_v10.py:23) → [run](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/orchestration.py:516) → prepare/教师准入/DAgger/PPO/export；阶段worker是内部工程工具，不是绕过门禁的正式入口。 | [全流程CPU分支](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_orchestration.py:103)；[教师失败不学习](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_orchestration.py:115) | 工程通过；真实全流程未执行 |
| §2 教师身份及学生来源 | [S2文件SHA/完整state加载](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/model.py:17)；[盲走SHA/normalizer](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/teachers.py:11)；[恢复也重新读取固定原教师](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/runtime.py:20)；教师路由使用同状态obs，不使用恢复学生作为楼梯教师。 | [真实源权重及Actor保持](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_learning_core.py:43)；[恢复教师身份](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_runtime_core.py:32) | 工程通过；能力准入未验证 |
| §2.2、§8 DAgger训练范围 | [D阶段仅Actor可训练](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/model.py:23)；[20/10逻辑更新、统一梯度分母](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/dagger.py:14)，全部更新完成后才入历史库。 | [微批/整批参数等价](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_learning_core.py:67)；[教师与学生参数隔离](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_learning_core.py:107) | 通过 |
| §3.1 机器人、控制与输入 | [复用名义S2配置](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/env.py:59)；[动作/关节顺序校验](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/env.py:63)；[展开机器人物理摘要](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/runtime.py:28)。控制0.02s、4×0.005s，动作q0+0.25a，原clip=None未擅加限幅。 | [机器人与地形身份分离](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_runtime_core.py:43)；physics_course_hashes.json | 工程通过；本轮未重跑物理 |
| §3.2–3.4 历史、地图和只读帧 | [45维真实上一动作及当前指令](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/env.py:159)；[每物理步只推进一次history/map](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/env.py:164)；[只读clone快照](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/env.py:180)；[复用40×30缓存、5步刷新](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v09/map.py:52)。无任务/教师编号Actor输入。 | [逐帧、跨reset旧新等价](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_runtime_memory.py:182)；真实GPU快照测试属于显式skip/历史短测范围。 | 工程通过；完整物理帧协议未重测 |
| §3.3 教师标注及动作帧 | [同一obs先student/teacher或Gaussian/value/logp](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/runtime.py:101) → [再执行物理step](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/runtime.py:115) → 复制原快照；[mask路由](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/teachers.py:27)不更新环境历史。 | [快照不被后续缓冲覆盖](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_runtime_memory.py:111)；新增SDS8.1跨教师日志回归 | 通过 |
| §4 几何与实际出生状态 | [薄踏面、有限地面、G/U和14m平台primitives](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/geometry.py:31)；[编译碰撞位置/尺寸核对](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/env.py:70)；[真实出生q/姿态/速度/历史/reset](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/env.py:120)；[同primitive最高面查询](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/geometry.py:49)。 | [G/U、接缝、范围](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_env_contract.py:8)；[病例实际初态不同](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_protocol.py:11) | 工程通过 |
| §5.1 指令、停车、重启 | [F_WALK/STAND1000步、F_STOP400步](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/tasks.py:55)；[100间隔减速](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/tasks.py:64)；[HOLD1000间隔、F_STOP重启250步](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/tasks.py:67)；平台按实际净位移5/10m。 | [逐帧减速/HOLD](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_env_contract.py:50)；[F_STOP五秒并非五米](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_protocol.py:83) | 通过 |
| §5.3 登顶和教师单向切换 | [四足平台内缩/足底高度/当前至少两足top force](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/env.py:243)；[整体两帧锁存、下一动作盲走、top_x锁存](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/tasks.py:47)；[超过平台前缘1m不切换即失败](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/tasks.py:52)。 | [两帧及同帧失败优先](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_env_contract.py:67)；新增跨30帧教师来源统计 | 工程通过；真实登顶能力未验证 |
| §6 接触、支撑与跌落 | [raw范数>5/reward OR上一raw/safe_support分离](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/contacts.py:10)；[同高顶面并集内缩0.042](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/geometry.py:80)；[route高度与分区走廊](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/env.py:236)；[真实接触/70度/跌落/越界失败](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/env.py:254)。 | [侧面接触不冒充支撑](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_env_contract.py:31)；[宽平台与route跌落](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_env_contract.py:23) | 工程通过；GPU保护测试本轮未重跑 |
| §7.1–7.6 六任务、地形与课程 | [角色比例和三速度](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/spec.py:13)；[D1/D2/D3组合生成](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/cases.py:13)；[P0/P1沿用D3地形](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/env.py:55)；[D1→D2→D3只在门禁后晋级](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/orchestration.py:449)。 | [实际分配及154/226/334组](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_env_contract.py:42)；[预算耗尽不越级](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_orchestration.py:121) | 通过 |
| §7.3 学习上限/晋级/保存 | [分阶段固定预算和P0额度](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/spec.py:21)；[charged须容纳完整rollout](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/runtime.py:197)；[跨2457600阈值保存、边界重合去重](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/runtime.py:211)；unused额度不转移。 | [精确轮数/门禁](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_orchestration.py:11)；[真实worker CPU跨周期保存/恢复](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_runtime_memory.py:64) | 通过 |
| §7.7、§10.4 rollout与bootstrap | [真实末观测bootstrap](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/runtime.py:128)；[terminal遮断GAE](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/ppo.py:7)；[完整rollout更新后保存/退出](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/runtime.py:201)；恢复为新物理回合而不冒充轨迹连续。 | [GAE边界](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_learning_core.py:23)；[partial不计完成更新](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_runtime_core.py:70) | 通过 |
| §8.1 每任务轨迹/失败/标签持久记录 | [完成长度/失败分母与未完gauge分离](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/runtime.py:75)；[坏标签形状或finite先工程报错](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/runtime.py:104)；[动作前teacher来源计数](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/runtime.py:117)；[worker实际写per_task及rollout/session ID](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/runtime.py:206)。 | [真实collector→worker出口、60步回合跨48步边界与恢复](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_runtime_task_logging.py:55)；其余PPO不适用及坏标签三变体 | 本次缺口已修复并通过 |
| §8.2–8.4 聚合与随机状态 | [每任务512=current256+history256及补足](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/replay.py:55)；[保存采样计数/RNG](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/replay.py:66)；[12×完整逻辑批分母与一次clip/Adam](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/dagger.py:23)。 | [六任务均衡](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_learning_core.py:17)；[改索引不改样本/RNG](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_runtime_memory.py:124) | 通过 |
| §2.3、§9 P0/P1与保持项 | [冻结T_keep、独立CPU种子Critic、输出零/std0.10](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/model.py:30)；[5epoch×4无放回批](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/ppo.py:34)；[仅P1 policy/entropy](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/ppo.py:44)；[六任务各256用T_keep重新标注](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/ppo.py:55)；统一clip后Adam/std投影。 | [P0隔离/重新标注](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_learning_core.py:115)；[全梯度微批等价](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_learning_core.py:142)；epoch_exact_once | 通过 |
| §10 奖励真实数据来源和事件帧 | [动作前action/phase/command/g_level锁存](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/env.py:197) → [4子步物理](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/env.py:210) → [任务事件及独立质量判定](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/env.py:266) → [真实joint限位/q/qdot/接触/高度组frame](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/env.py:285) → [实际调用v10 compute_reward](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/env.py:298)。 | 旧20项手算、失败优先测试；本轮另独立CPU核对巡航/停车20项实际权重及dt。 | 通过（工程及CPU代数） |
| §10.3 真实稳定、抬脚与腾空 | [速度/姿态/净空/四足安全支撑/足速判断stable](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/env.py:252)；[真实足下表面判stairs](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/env.py:248)；[reward接触计时/初始下落抑制/停车清零](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/rewards.py:12)；[实际软限位](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/rewards.py:29)。 | [全部raw项手算](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_learning_core.py:97)；[腾空与停车清除](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_learning_core.py:33) | 通过 |
| §11.1–11.6 固定病例与逐例质量 | [实际初态和suite SHA](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/cases.py:60)；[逐例完整窗口和不可逆姿态违规](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/evaluation.py:76)；[完整唯一分母、绝对/相对阈值与覆盖率](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/evaluation.py:149)。 | [重复病例拒绝](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_protocol.py:40)；缺窗口/NaN/端点/漂移起点回归 | 工程通过；正式成功率未验证 |
| §11.7 A后B/独立终验 | [固定同SHA A→B，中间不训练](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/orchestration.py:370)；[终验一次开封](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/orchestration.py:390)；[DAgger验收后PPO，已通过PPO门禁恢复直接终验](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/orchestration.py:471)。 | [终验失败不越级](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_orchestration.py:128)；[恢复不泄漏终验后训练](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_orchestration.py:237) | 工程通过；未签真实行为证书 |
| §11.8 PPO回退 | [连续保持失败/最多两次回退](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/orchestration.py:74)；[恢复最近通过或P0，事件原子记录，不改LR/奖励/keep/std](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/orchestration.py:502)。 | [回P0、不退预算](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_orchestration.py:138)；[第三次停止](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_orchestration.py:19) | 通过 |
| §12、Q1 样本账本与恢复 | [step前reserve/执行后mark](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/runtime.py:112)；[单在途及硬上限](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/checkpoint.py:179)；[uncertain不退](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/checkpoint.py:196)；[拒外账落后于checkpoint并恢复学习状态](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/runtime.py:158)；partial只收费用不伪造更新。 | [崩溃/重复恢复/边界](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_runtime_core.py:18)；[旧checkpoint不退款](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_runtime_core.py:5) | 通过 |
| Q2 评估首终止/padding/资源 | [同max_steps批](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/eval_worker.py:71)；[全批物理计费](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/eval_worker.py:103)；[仅active推进独立metrics、首次终止冻结](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/eval_worker.py:108)；[独立进程及资源通过后发布成功receipt](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/eval_worker.py:144)。 | [费用上界等式](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_protocol.py:72)；GPU冻结padding测试未本轮执行 | 工程通过；真实完整评估未执行 |
| Q3 文件身份/证书/回退状态 | [schema/SDS/code/教师/N/physics验证](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/checkpoint.py:94)；[事件引用SHA及单调事实重放](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/orchestration.py:141)；[证书绑定原checkpoint/结果/suite/规则](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/orchestration.py:434)；[事件→controller原子状态](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/orchestration.py:180)。 | [原文件不改/篡改拒绝](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_orchestration.py:164)；[事件后崩溃幂等](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_orchestration.py:179) | 通过 |
| Q4 保存/释放/真实加载 | [完整写盘、流式SHA及CRC校验](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/checkpoint.py:77)；[保存后释放旧replay再实际load](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/resources.py:103)；[正式恢复同ownership转移](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/runtime.py:159)；[take清解码源引用](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/replay.py:76)；[全程stage/后台峰值](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/resources.py:43)。 | [完整3GB CPU原始8pass证据，未重跑](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_runtime_memory.py:139)；保存、整数、sampler、source保留回归 | 工程通过；真实资源复测未做 |
| Q4-A 次完整栈准入与全局硬cap | [序列仅要求下一1024完整栈](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/resources.py:151)；[parent逐stack独占容量](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/resources.py:180)；[child同scope复用lease](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/resources.py:60)；[只有实际step增charged](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/probe_budget.py:311)；[只释放未执行余量](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/probe_budget.py:281)。 | [A实际数字、fakechild不启动未足额512](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_probe_budget.py:194)；[幂等/互斥/uncertain不退](/home/xxs/dog/unitree_rl_mjlab_a2/tests/test_v10_probe_budget.py:246) | 本次入口缺口已修复；执行审批仍阻断 |
| §13.4 单学生导出 | [验收证书/原checkpoint核对后导出Actor](/home/xxs/dog/unitree_rl_mjlab_a2/scripts/run_a2_v10.py:43)；[正式PPO通过后才调用export](/home/xxs/dog/unitree_rl_mjlab_a2/src/tasks/velocity/v10/orchestration.py:526)；契约明确map/history/route及无teacher运行时。 | CPU控制流覆盖；没有真实已验模型可执行最终部署验证。 | 实现接线通过；实际导出/播放未验证 |

## 奖励接线的独立核验

本轮重点检查的是实际 `V10Env.step` 的参数接线，不只检查数学函数存在。动作执行前锁存phase、command、g_level、previous_action及x；真实4子步后取得速度、姿态、q/qdot、PD目标、接触及真实表面；先物理失败、再任务/质量事件；组frame后实际调用v10奖励，最后才推进下一观测或reset。因此登顶奖励属于导致事件的transition，教师和停车指令影响下一动作，同帧失败屏蔽top/success。

实际总式为 `r=0.02Σ(w_i f_i)+0.4Δx·I_cruise+2I_top+8I_success−10I_failure`。与旧代码把HOLD零指令冒作stable、软限位填零不同，当前stable来自真实世界速度/姿态/净空/四足安全支撑/足速，软限位来自真实q及硬限位90%范围。接触计时明确采用C_reward，不能宣称与旧raw计时完全相同。

审核者本轮另执行一次轻量CPU代数核验，无GPU、无环境物理、无新增测试文件：对巡航和停车两相位分别逐项比较20个实际weighted项与SDS常数×raw×0.02，验证角速度相位权重、腾空关闭、stable、净位移不再乘dt；全部匹配。混合接触输入下slip与clearance均非零，避免对应权重被零输入掩盖。原有全部raw手算/事件优先回归继续作为公式证据，不把这项代数核验当作策略效果测试。

## 实际证据复用及未验证项目

- 基准同源码原始日志 `logs/a2_v10/q4_all_v10_final.log`：83 passed、5 skipped、1上游Torch JIT警告、81.97秒。已通过19模块字节比较确认未改部分可复用；它不是新增A/遥测功能的覆盖证明。
- 完整196608条真实形状FP32、约3.10GB checkpoint的CPU保存/释放/实际加载与model/Adam/normalizer/RNG/下一批等价：原始 `logs/a2_v10/q4_cpu_memory_test.log` 为8 passed、1 warning、69.18秒，配套 `q4_cpu_full_replay_memory.json`。这些路径本次未变，未重跑昂贵完整内存测试；该证据物理样本为0。
- 本次定向修复原始日志 `logs/a2_v10/final_acceptance_gap_tests_20261127.log`：24 passed、1 warning、28.65秒，包含A/lease17项、真实collector→worker日志5项及身份2项。SDS8.1独立RED/GREEN保留在同目录，GREEN5 passed。CPU fixtures的合成费用不计作真实probe费用或行为成功。
- 实现者对A/lease+既有controller报告37 CPU passed；该数字来源为实现者tool输出及 `logs/v10_orchestration_implementation_report.md`，本轮没有重新制造一份所谓“独立37项原始日志”。独立结论主要依赖实际接线复核及上述真实24项原始输出，而非盲信“已通过”摘要。
- 本轮只读校核和CPU奖励代数检查均未启动V10Env、probe、训练、教师suite或GPU。没有新物理样本。

| 验收层 | 截至本报告的实际状态 |
|---|---|
| 工程代码/配置/恢复/预算/接线 | 本次两项缺口修后通过，绑定本报告的最终源码指纹；尚待主对话提交推送修复和文档 |
| 原资源探测 | **失败**：1024 DAgger98304、512 DAgger638976条；512最低MemAvailable2883108864<4GiB。不能把完成25轮与restore_checked当作资源通过 |
| 新A范围真实资源复测 | **未执行**；代码路径已验证，执行授权审批另有阻断（见下节） |
| 教师能力准入 | **未执行/未验证**；文件身份与同状态适配通过不能代替≥18/20能力结果 |
| D1/D2/D3、独立DAgger终验 | **未执行/未验证**；正式学习0，无真实V10_DAGGER_ACCEPTED |
| P0/P1、独立PPO终验、部署播放 | **未执行/未验证**；无真实V10_DAGGER_PPO_ACCEPTED、无可声称的效果改善 |

## 当前预算和自动审批状态

真实全局probe历史费用仍为charged=executed=737280、updated=663552、uncertain=0，保留在 `logs/a2_v10/global_probe_costs.jsonl`，迁移并不产生新物理费用。旧报告“尚待预算答复”是当时时点的历史事实。

2026-11-27规划线程转述的A方案把累计charged硬上限设为3833856，保留历史后余3096576；工程代码已按该方案逐栈准入、独占容量、逐step计费，不要求一次付足整个最坏回退序列。若1024两栈各用完，CPU回归示例的累计为3293184、余540672，下一512完整栈需638976、缺98304时不得启动。这里的3293184是临时CPU测试账本示例，不是真实生产账本新费用。4571136不是本次A上限，学习/评估不转借。

**执行审批尚未解除。** 主对话尝试创建A生产预算授权文件时被自动审批拒绝， 说明的原因为可信用户原始消息未明确3833856具体数值，不能仅凭助手/规划转述扩授权。主对话未绕过、未创建可执行生产授权文件、未执行新物理。此事实不阻止签署工程验收，但工程通过不能替代可信数值授权；本报告不代写授权、不授权立即重probe，也不误写成继续等待B方案增额。

主对话取得有效执行授权后，仍必须依次通过资源双栈、教师A/B、DAgger课程/独立终验、PPO课程/独立终验，才可交付行为验收通过的统一学生。全部条件未满足前不能声称“A2 v1.0训练/整体目标已完成”。

本报告是唯一新增文档，既有报告、已有未提交工作及源代码均未由审核者改写。主对话负责追加 `logs/codex_changes.md`、提交推送并核对远端，本报告不冒称这些后续动作已完成。
