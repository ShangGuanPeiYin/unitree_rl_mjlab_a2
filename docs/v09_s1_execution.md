# A2 v0.9 S1 执行记录

规范为SDS v1.1 / `v09-native40x30-sequential-r2`，SHA256为`500ad32aaa946c927459427f3a65565b15d4b77fca4be214817121d15c742976`。规划补充裁定Q001～Q010保存在`doc/sds/reviews/v09_execution_rulings.md`。

源权重为`logs/a2_ideal_stairs_v07/continue_seed42_from004817_trial01/checkpoints/stop_006400.pt`，SHA256为`30ed23ba16f7e490afca001fbbecadb4a11ff4b11733a5ae7ac7d0c841f9a8b2`。执行目录为`logs/a2_v09/s1_seed42_r3`，run_id为`a2-v09-s1-seed42-r3`。早期r1/r2冻结及其批准保留为零物理消费历史，实际执行仅使用r3账本。

## 流程与训练设置

S1.0保存6400在全部75组的A/B及三套终验对照，并用独立训练校准集冻结值尺度。S1.1完成地图分支适配，再由S1.2恢复完整原能力。通过同一候选SHA的相邻A/B和当前独立终验后才能生成`S1_RESTORED`。

| 项目 | S1.1 | S1.2 |
|---|---|---|
| 学习样本上限 | 20,000,000 | 主段630,000,000，终验失败恢复最多两次各100,000,000 |
| 可训练部分 | Actor/Critic地图编码器、Actor融合层 | Actor均值网络与Critic全部 |
| 教师控制回合比例 | 50% | 10% |
| 学生控制回合比例 | 50% | 90% |
| std与归一化 | 继承6400并冻结 | 继续冻结 |
| Adam | LR=3e-5，betas=(0.9,0.999)，epsilon=1e-8，weight_decay=0 | 同一优化器与已有动量 |

训练seed=42，rollout=48控制步，每次5epoch×4逻辑minibatch，共20次Adam。微批次最大512，按逻辑批总样本数累计梯度，再裁剪到1.0并更新一次Adam。控制周期0.02秒，物理周期0.005秒，每控制区间4个物理子步。巡航速度0.3/0.5/0.6m/s，vy=wz=0。环境分配为普通平地走20%、平地走停20%、H00～H15合计30%、D25/D22/D20/L12/L20/L40合计30%。

S1学习使用动作均值MSE及归一化Critic值MSE。

\[
L=\frac{1}{12}\sum_j(\mu_{09,j}-\mu_{6400,j})^2
+0.1\left(\frac{V_{09}-V_{6400}}{s_V}\right)^2,
\quad s_V=\max(\operatorname{std}(V_{6400}),1).
\]

s_V由600个独立校准病例的首次物理回合值确定，使用FP64累计总体标准差，最多945,800个实际控制样本的冻结清单、预留上限1,000,000。原始值及病例slot保存为可重算证据。实际冻结s_V=2.136540383972894，来自289,564个首回合value；21,588个等待样本未进入统计。独立FP64重算已通过，原统计值及原文件保持不变。

## 地图和网络继承

地图由3×48×32改为原生3×40×30。覆盖前1.2m、后0.8m、左右各0.75m，间距0.05m，10Hz刷新。两个地图FC从64×768变为64×640，按六个旧长度位置到五个新位置的区间重叠系数迁移。其余51项状态精确复制，保留路线旁路、5×45历史与35维Critic特权量。教师和学生对同一执行状态分别真实查询旧、新网格。

## 门禁与证据

每套75组包括平地行走3组、平地走停6组、22类旧楼梯×3速度66组。A/B各1500例，终验每套7500例。楼梯保持成功数、终止净距离P10/P50/P90和配对均值、各类物理失败数量。平地按Q001逐指标匹配教师完整窗口病例，报告固定分母、有效覆盖、缺失原因、均值或中位数及Wilson95区间。

S1.2主段每1亿学习样本及末尾检查，阶段最多11次门禁；相邻A/B必须来自同一冻结候选。按Q006/Q007记录实际attempt、原计划slot和取消阈值。最终集最多三套，恢复和预算不因checkpoint回退。

每500万学习样本保存checkpoint，保留最新5份，并保留课程/阶段节点和每1亿归档。schema11保存模型、Adam、RNG、角色和源身份，外部只追加账本保存累计费用及门禁/最终集/回退状态。恢复采用新物理回合与单调episode ID。

独立S1_DIAG前后各1500例，使用同一批保存物理观测和6400标签比较两个模型损失，并报告各自实际轨迹。视频固定DIAG每组case_id=0，共75例，包含失败病例和reset前真实终止帧。

## 资源及预算

按2048→1536→1024→512实测，选择首个稳定档。每档5次热身、50次测量，每次完整48步与20次Adam，另包含全参数S1.2更新、卸载、独立100环境评估、独立1环境录像、退出后恢复更新。要求整卡峰值≤7168MiB、主机可用内存≥4GiB。按互斥资源布局分别取峰值，避免把已卸载训练与子进程峰值相加。

| S1非学习预留 | 控制样本上界 |
|---|---:|
| 原始6400五套对照 | 40,196,500 |
| 校准预留 | 1,000,000 |
| 候选门禁与三套终验 | 61,477,000 |
| 四档S1_RESOURCE | 16,030,340 |
| S1_EVIDENCE诊断与视频 | 5,438,350 |
| 合计 | 124,142,190 |
| 距150,000,000非学习上限余额 | 25,857,810 |

以上为最坏预留，实际按推进的所有环境slot计费，包括已完成slot等待。未知是否执行完的操作按预留上界保守结算，保留状态与异常原因。父子进程同一操作只结算一次。

## 当前验证状态

已完成CPU模块与流程测试、源6400/SDS哈希校验、实际compiled地形231个geom一致性及1mm错位拒绝、真实CUDA/NVML零物理监控、4帧640×480/25fps本地FFmpeg编码和完整解码。最终源码清单及各次独立批准保存在执行目录。

2026-11-20，原6400的gate_A已完成75组×20例（1500例），成功1051、失败448、超时1，实际控制样本817,400，该义务已结账。完整组表见`docs/v09_s1_teacher_gate_A.md`。gate_B也已完成75组×20例：成功1037、失败460、超时3，实际控制样本869,800，义务已结账；组表见`docs/v09_s1_teacher_gate_B.md`。A/B合计1,687,200实际控制样本。两套均已通过独立全量CPU证据复核，分别见`docs/v09_s1_teacher_gate_A_independent_review_20261120.md`与`docs/v09_s1_teacher_gate_B_independent_review_20261120.md`；此结论不批准资源或学习。教师final_1缓存已采集7500例并结账，实际4,760,300控制样本，成绩继续封存，75批/7500例已通过独立完整性复核（非行为验收）。教师final_2也已采集7500例并结账，实际4,327,600控制样本，独立完整性复核通过，成绩仍封存。教师final_3已完成7500例并结账，实际4,644,400控制样本，独立完整性复核通过；三套final成绩仍封存。五套教师对照共25,500例、15,419,500控制样本。600例校准、旧finalize及依赖连续性已独审通过，见`docs/v09_s1_0_sealed_integrity_review_20261120.md`。冻结s_V=2.136540383972894，289,564首回合value、21,588等待样本不入分母；校准311,152控制样本。S1.0总15,730,652控制样本（62,922,608物理子步），外部ledger闭合seq907，正式学习0。baseline_admission已生成且baseline_complete=true；resource_admitted=false、behavior_restored=false，Q010复用和资源/学习仍未批准。

新增独审`docs/v09_s1_resource_evidence_review_20261120.md`确认资源CLI将20例门禁误比较为100槽的入口阻断，尚未批准资源执行或学习。按规划Q010，先完成旧447全部S1.0并封存，之后再修入口并逐项审核旧证据向新源码的单向复用；旧生产者SHA不得改写。真实资源负载、固定机器人视频、S1学习和行为恢复仍未验证；正式学习0，S2/S3尚未启动。


## Q010限定修复进度（2026-11-20）

旧S1.0已完整封存后实施消费端修复：资源入口严格验证gate_A/L40/0.3的20个唯一源病例，再复制五轮形成100物理槽；每槽5005控制步和录像1槽5005步的资源负载保持不变。共享EvidenceIdentityContext仅允许独审清单中此次旧447→单一新源码的原文件，保留原producer与内容SHA；新报告、学生checkpoint、DIAG和学习批准仍严格绑定新源码。封存final原文仍须真实当前候选及当前套资格，不因hash核验解封。

上一候选231项CPU回归通过后，fresh独审发现资源子CLI同时省略两身份参数能绕过清单字节校验（R-RES-ID-OMIT01）。该候选未获批准；已用evaluation/video两模式及gate尾加空格的真实CLI测试复现，已限定修复，资源CLI 30项CPU测试通过。新源码c9ee8789f6051bc6c36cbfc24f076deba6121d8ffedb48ce5d689e0a6b665ea4已冻结，全部16模块239项CPU回归通过（197.04秒，源/测试前后hash一致）。独立代码审核通过，报告见`docs/v09_s1_q010_code_review_20261120.md`，独立限定62项通过；代码ready不等于实际兼容或资源准入。协议独审继续核对旧证据及运行依赖，之后签发直接复用清单，再使用真实校准做零物理dry-run；资源工程入口独立批准且实际资源通过后才另审正式学习。当前无真实资源执行或S1学习，未验证行为恢复，S2/S3未启动。


Q010协议独审已通过，见`docs/v09_s1_q010_compatibility_review_20261120.md`。正式只读清单`q010_s1_0_evidence_identity_c9ee.json`（SHA `0d158b266dde7daeefaf81bc5b270a4294a94f473203015ea8fd9d6df6380408`）位于r3目录，且在源码snapshot之外，唯一scope为S1.0_evidence_reuse，绑定7冻结输入、12必要runtime及931仅hash审计材料。旧原件和费用不变；它不批准资源物理执行或学习。

使用同清单及真实旧校准的资源主CLI dry-run已exit0，20源/100槽×5005及录像1槽×5005校验通过，活ledger字节不变，无输出目录、无物理执行。本机证明`q010_real_resource_dry_proof.json` SHA `df567e11c48d8408758e3a3361ce91c9f59c5521c1a8ca92a0ebf3314b5de5c1`保存完整显式命令及来源引用；大stdout留本机。独立S1_RESOURCE工程入口已批准，见`docs/v09_s1_q010_resource_entry_review_20261120.md`；须完成任务提交/push及远端核验后才执行同一干跑命令加--execute。批准仅工程探测，resource_capacity_passed=false、learning_approved=false；实际资源及S1学习尚未启动。


## Q010真实资源探测结果（独立容量验收通过）

已审24文件提交`a43b48eacd32b6c8f578a3a3206d41fbb5896c60`非强制上传，启动前再次核远端一致。一次性探测exit0，程序选择512；本机报告`resource_q010_r1/resource_admission.json`语义SHA `d712183d48eaa89db4c76209cbf4e52b99d8bd83d873b724a3e409c599e17f3b`。

| 环境数 | 观测整卡峰值MiB | 准入保守峰值MiB | 结果 |
|---|---:|---:|---|
| 2048 | 8105.5 | 8827.5 | 热身1次后下一迭代OOM |
| 1536 | 8117.0625 | 8169.125 | 热身2次后下一迭代OOM |
| 1024 | 6953.1875 | 9635.3125 | 完整负载完成，保守峰值超过7168上限 |
| 512 | 5167.625 | 6367.625 | 完整负载及独立容量验收通过 |

1024与512均完成5热身+50测量、全参数S1.2更新、独立100环境×5005评估、独立1环境×5005视频及恢复更新；两视频验证和reset前终止帧标记为true。512最低主机可用内存11.028236389160156GiB。保守估计与采样观测值分列，未调整原7168MiB/4GiB门槛。探测模型全部为一次性工程模型，正式学习不继承其权重或Adam。

资源收费5,631,298控制样本，其中已知完成5,459,266，两个OOM未知操作按预留上界收费172,032。活ledger seq1157闭合，各资源操作无未结算；S1累计收费21,361,950，正式学习操作和样本均0。上述总收费不能全部表述为已知实际物理步。

真实train CLI无execute预检通过，账本未变、CUDA禁用：512环境每迭代24,576样本、20次Adam；S1.1最大813次（19,980,288样本），S1.2主段最大25,634次（629,981,184），终验行为失败后每次恢复最多4,069次（99,999,744）。seed42、rollout48、5epoch×4逻辑minibatch、microbatch512、LR0.00003、clip1、每5,000,000学习样本保存及原sV均不变。只读预检证明`q010_s1_preflight_proof.json` SHA `54e5b1210d433df7fd76d0a883a32cda5351f2e15b55ed9fffb9897562cb6a32`；它没有学习批准。实际资源容量已独审通过，正式学习准入仍另审，行为恢复及S2/S3均未完成。


独立容量验收正式通过，报告`docs/v09_s1_q010_resource_capacity_review_20261120.md`（SHA `2b8882c71194c85c6dc32e6c78ea4c5bb1801d224491da7b595dc25a73586f76`）及本机proof（SHA `b6bc45b9e6fcf92b79884085cfd6154afd12148fcff9b095b7d6f6256d8aac65`）独立确认最大可准入档512。两档完整运行视频各2505帧已完整解码；18个地图Adam状态从1100至1120步，新增26个全参数状态各20步、共44状态，冻结统计/std及RNG/对象身份恢复证据通过；59工作项和1157闭合收据一致。验收者未新增物理，没有解封final成绩或批准学习。S1.1/S1.2学习作用域及具体命令继续单独审核。进度记录提交`651f820b991ce517775a83a31716d2bd291ebf1a`已非强制上传并核远端一致，源码仍为c9ee。


## S1学习独立准入（已签发，尚未启动）

独立报告`docs/v09_s1_q010_learning_admission_20261120.md` SHA `e2dd965924f1a86c14a2526cbf2db75f9c01ab34fdc4fae86f7a7cca2b3f57f3`及本机批准`s1_learning_approval_q010_c9ee.json` SHA `9d4c515d799ecfbd586e57ba7ec75d436e610dca4738a9ab1086ccb2580d689f`只授权S1.1、受运行时工程条件限制的S1.2及既定串行证据/门禁/终验子流程。绑定同一C9源码、r3、6400/SDS、0d158旧证据清单、512容量报告；没有S2/S3执行资格或行为恢复结论。

独审实际CPU admissions/schedule/最终learning approval验证通过；root再次真实构造同Context并验证最终JSON、报告/清单SHA和source/test freeze一致，通过。账本未变，正式journal不存在。唯一允许命令为已保存真实preflight.command追加--approval-report本JSON及--execute，CUDA_VISIBLE_DEVICES=0，无resume；禁止使用probe权重。

提交报告、日志、计划并核远端后才启动。首次正式学习rollout之前，驱动先做75组×20例的S1_DIAG初始诊断、6400与迁移初始各75个固定视频；813个完整适配rollout后再做同套诊断和75个适配后视频。有限性、迁移/几何/接触一致性与完整对照条件满足后才进入S1.2；其余行为门禁、预算和final鲜活资格保持原合同。当前仅学习准入通过，尚未推进上述物理或正式学习，效果未验证。


## 正式驱动学前阶段（2026-11-20）

提交`2e532609c83fc75197f7abf9cd0184eb9a0dbcbc`已上传并再次核远端一致后，按唯一批准命令启动PID3228141，运行日志`q010_s1_formal_20261120.log`、启动收据`q010_s1_formal_launch_20261120.json`留r3目录。首个学习rollout前正在执行Q008 before；启动不表示已有梯度学习或行为改善。

正式初始checkpoint文件SHA `32c962f28352b13f81c1c9b39e4c5a9bbe01f5e3f6545d98f8b46414492e6c13`，卸载前checkpoint文件SHA `4893c425476368592488bde8e155117194178e0d266d8438bdeb5f5be09f3042`；两者模型state SHA相同 `f9a038b73bf2e97290cfb38428d9a6d186e1a136f908ec664e7a0c2ebc8d853c`。独审本机proof `q008_formal_initial_checkpoint_independent_proof.json` SHA `a5a802c78e6ecf5f708baf992329f02d33069ffdd782a4420743ebb42ae2eba1`确认53条state中51条非变化参数/buffer与唯一6400逐张量精确一致，另两个FC独立Fraction重叠系数重算也精确一致。std/stats精确继承、18参数可训练、Adam管理45参数且状态为空，同Context、learning_samples=0；没有加载旧或probe训练状态。

上述仅正式初始化工程证据，before诊断与两模型各75视频尚在采集，完整性待闭合后独审；S1.1梯度学习和后续S1.2行为效果均未验证。


## Q011首梯度阻断（数值核验进行中）

初始DIAG1500例已闭合539740控制样本，但CPU重算与保存GPU教师标签有原容差例外，尚未整套验收。规划给定同GPU两次零物理重放及56个active子集的固定N20补位/遮盖控制合同，原标签、数据、容差和C9源码不变；详见`docs/v09_s1_q011_numeric_audit_20261120.md`。只暂停Q009已卸载且do_wait的父PID3228141，teacher录像子3239514继续自然闭合；学习仍0。GPU重放等待子进程退出后串行执行，独审通过Q011前不CONT父；不能将CPU TF32反事实当GPU因果结论或已恢复行为。


Q011 v3离线工具经fresh复审两Important闭合，CPU独立17项通过后，root复验同PID/starttick/T态、journal环境/双缓冲已释放、录像子已自然exit0且无CUDA上下文、live1464无pending后，串行执行两个独立GPU进程。safe proof SHA `1e02d8dfff8ee5145e13041cf78693c0abc1447d6f63a0d3f435e50ff16d9f09`。两个进程3276157/3276203各520组：342完整批（340异常+2对照）、56部分×两填法、33 mask×两填法；全部原tol对保存标签、两次互比、填法和遮盖对原full参考均通过，数值最大差均0。原件/容差/flags不改，56部分只证明保存active，不称恢复历史inactive。

程序exit0、0环境物理步、账本SHA `9478b5ef2ac71388872d6b0272f84a393bcbd97ba98db458e0378a1862ff44de`未变，报告文件SHA `f657e49d9d93ee45879646ef70ef8188dc4515aaf1b926ed4978f5000328425a`。程序不自动解Q011，父仍T等待实际输出独审签字；CPU旧误差和TF32反事实限制保留，不能据GPU通过断言具体GPU算子因果。


## Q011独立解除（恢复原父进程前）

独立结果报告`docs/v09_s1_q011_numeric_audit_result_20261120.md` SHA `50f8d46a535255537fbc2a1911dd78d04a19f21659afd919a54f3d84878a7e3a`及本机clearance SHA `22139fcb8ce685adb7191c67cc8ed6a35822607d04bf35b35185bb1cf1edbfe3`签发，只解除Q011，不是新学习批准。独审从实际两PT独立重算221728个saved分量和全部重复/双fill/遮盖参考，原tol全部数值差0、0例外；模型、时序、槽位、软件/precision/layout及原数据SHA核对通过。旧CPU差异及不可历史重建inactive/未直接记录历史flags/未证明具体算子因果限制全部保留。

独审证明`q011_actual_gpu_independent_proof.json` SHA `f67ae1ffd3719b1b5898cff4a16c59f69f53c33f4e0913ce30e390d39d69b020`，GPU重放物理/收费0，live1464/file9478...未变，学习仍0。原9d4学习批准真实同Context复验有效，clearance因只有Q011 scope被学习validator拒绝，不能冒充新学习授权。

root先将裁定、结果和日志提交push并核远端，再核原PID3228141/start_ticks145922371/完整argv和账本，仅CONT一次；不重启、不载入probe、不回退费用。随后仍需init75视频及既定全部Q008条件，不能据数值标签通过宣称训练或停车效果改善。旧裁定doc、所有既有签字文件、C9源/tests/catalog保持不变。


Q011解除记录提交`fbcc118cb89dc0ddfb3f1cdac424c47a85ebc30e`已push并在恢复前再次核远端一致；原PID3228141/start_ticks145922371/argv/账本重核后仅CONT一次，同一父进程已继续init75视频。append-only resume收据SHA `9d493388cae0eb41950f5e07b173bc0014fb2f1bf9834c4017f8ab7f98fc7a79`，恢复后独立核teacher旧子已回收、teacher仍精确75操作/35954收费不变、init已启动，post-resume proof SHA `1cdae128c64d1ddeca10f2aa88be92ba7304b50abf9c55f2e6a8689577b5cf3c`。

6400固定75视频只CPU独审已通过，proof `q008_teacher75_videos_independent_proof.json` SHA `f6c7799195f09c7170f5ae2b4e107901053a075db992454d2492dee26089e1bd`：75 case0/manifest/C9/Context/源6400一致，全部独立完整解码共18070帧、640×480/25fps，初始帧/每2控制步/精确首终止帧及Q002命令阶段均核对通过。费用35954与75项已知complete及closed义务一致，无unknown或重复费用。初始迁移75视频尚采集中，尚无完整学习迭代，学前Q008整体不能提前宣称通过。


## Q008 before完整独审通过，S1.1运行中

完整报告`docs/v09_s1_q008_before_evidence_review_20261120.md` SHA `b4304067051fa649b4c30d78de8813777b12ee3fa3de288474431cc627a50d0b`，汇总proof SHA `10295a9b4e10d58197358117b9e73667f5c91d7a536b48826543b65bd19917ca`。初始化53-state迁移、DIAG1500例/75PT与几何、Q011GPU数值链、两套150视频工程完整性均通过。新增init75 proof SHA `9aea51581788224c72a8b678ba7f29542bdf2ac49318f3c43cc712e13d26eb78`；仅新增75做了完整解码，未重复teacher75。

学前费用DIAG539740+teacher视频35954+init视频21605=597299控制样本，全部known complete，无unknown/重复；对应2389196物理子步。两套视频共28965帧（teacher18070、init10895）。权威账本closed次序diag seq1313、teacher1464、init1615，均早于首学习reserve1618、commit1623；journal before seq5/恢复6/首optimizer8，首轮24576样本20Adam有限。Q011等待与GPU核验没有新增物理费。

S1.1实际训练已开始，本次记录时确认97/813完整迭代、2383872学习样本，最近20Adam及loss/梯度有限。训练继续原N512/48步/5epoch×4逻辑批/micro512/LR3e-5/clip1/std及normalizer冻结；每5m学习样本保存。before工程通过不代表loss改善或恢复，after1500例/75视频及同数据比较仍是S1.2工程条件，正式A/B与final另验。


## 首个500万学习样本checkpoint独立工程审查通过

报告`docs/v09_s1_checkpoint_005013504_independent_review_20261120.md` SHA `74ee0915599e16e84bca7718253cb04282a8554286c82861e0967911bee3d8fd`，本机proof SHA `4bd47530f7df414581f441e6a1b28244aab9eec987270e1559954bb8fa8bac97`。指定首周期文件SHA `e43257834144000f5f8b810fcd7539c1b9d04333bee4098226f9d95d6ab327a1`，在第204次完整迭代后保存，对应5,013,504学习控制样本；第203次为4,988,928，故符合每5,000,000学习样本在完整rollout边界保存的原规则。文件名415为journal保存事件序号。

纯CPU独审确认同C9/SDS/6400/Context，实际初始checkpoint的53项state中仅18项学习分支变化，其余35项精确一致，包含std与8个normalizer buffers。Adam管理45参数，恰18份状态各4,080步=204×20，模型及动量有限，原LR3e-5等设置不变。CPU/Python/NumPy RNG局部装载通过；CUDA/shuffle/environment的16字节编码完整，未额外执行GPU恢复或物理重放。环境offset0由实际named-stream病例及initial_cases源码解释为合法未消费，不是训练异常，未修改生产数据或设置。

保存前缀seq2443、7,737,998字节、SHA `18d57dfb84ff748862589b3848c6dd5e684d52655d9198573fae7ee7bb40a372`，204项学习操作完整结算，metadata阶段/累计费用及角色租约精确对齐；活ledger后续合法追加，未回退消费。工程一致性通过不代表行为改善、S1.2资格或S1_RESTORED。S1.1继续既定813迭代上限，after1500例诊断/75视频/同数据比较以及A/B/final仍按原条件执行。没有修改冻结源码/tests、签署旧报告/catalog、训练权重或运行配置。


## S1.1完成与Q012 after数值待裁定

813轮/19,980,288学习样本实际完成，适配checkpoint SHA `3453350b921e81d0e3e568a3d6c93a6044b8c6b62edc6d3362083a0a0235d800`；CPU独审proof SHA `3401b0d5502db9674530d24f8e1609782931dd69cd308c33d6fe376da6abf74a`确认35冻结项精确一致、18Adam状态各16260、全部813轮有限性及账本/journal关联。after首已结算批400状态的4800动作分量中162超原tol，value0/400超差，raw指标/geometry/finite通过；不扩展before Q011结论。详见`docs/v09_s1_q012_after_numeric_audit_20261120.md`。

为阻断未闭合数值验收下的S1.2，原父同PID/tick/argv与Q009 after已卸载/do_wait身份复核后，只SIGSTOP父一次，子diag_after继续自然采集。hold receipt SHA `54361dbab6e232b075611705c8afe9d7826cef1f85ce9f1819a1d9a2526479b3`。Q012消息两次被自动审核拒绝，查到历史规划用户明确回问授权后仍未获通信审核放行；消息未发出，当前用户发送许可异步待答，不通过其他渠道绕过。新GPU核验/恢复未经裁定不执行；after整套、比较、视频及S1.2资格仍未验收，不作行为恢复结论。


## Q012数值独立解除（恢复原父前）

独审报告`docs/v09_s1_q012_numeric_audit_result_20261120.md` SHA `1847986dcdf86fe9c6eb53a15defed08eb167e82c7d4127d386471c39feda524`；本机clearance SHA `8701392499df92ccdbc5efda9fd265543047f3938cf6a45155bbfa78f11d3560`，actual proof SHA `691a781b94e0ba0cb32d9e58a1e0893f857b9b6e69ddb514eefb0bdd2d85d5b0`。独立CPU从双PT重算1130操作/各进程teacher μ196680、V16390及student μ18000、V1500，saved/互比/241双fill/218mask控制均max/RMS/超差0。全109实际mask、75学生及原件/软件/精度/槽位/冻结状态通过，物理与新增学习0，ledger5067/file369a…8730不变。原CPU例外及历史flags未直接保存、非真实inactive历史重建等边界保留，不断言具体算子根因或行为恢复。

原9d4学习批准同Context仍有效，Q012文件仅解除数值等待，被学习validator作为新学习批准拒绝。root提交push并核远端后复核同PID3228141/tick145922371/argv、子自然exit0及账本，只CONT原父一次；随后仍需production四路comparison/adapt75及既有S1.2条件，不能用数值签字替代。


Q012解除报告提交460922c236d199808b1de1924dffdd41fb628186已push核远端后，root复核原9d4同Context/C9及同父PID/tick/argv、diag子exit0、账本5067不变，仅CONT原父一次；append-only resume收据SHA `01c18bb3348b10fd632fbef0a0d599bc45dce026bee193dbc2d024b9f4d7d851`。原父继续production四路比较及adapt75，没有重启或改配方。

Q013参考口径见`docs/v09_s1_q013_loss_reference_ruling_20261120.md`，SHA `b4a9e1febb8040c2a2fa83ccda1ac90b17e515342fe107cafa430b0da3fe4b43`：GPU正式值与CPU参考分别保留，不新增跨后端聚合loss门槛；独核公式/归约/原sV/分母和有限性。comparison工程flags均true，完整独审及adapt75仍待闭合；S1.2尚未启动。


## Q008 after完整独审通过，正式S1.2运行

完整报告`docs/v09_s1_q008_after_evidence_review_20261120.md` SHA `73811ec6edbf393a9ed7d6f269a3e0f45c52c0ad77ceba748a20fd4a1ce090b0`，综合proof SHA `f83b1143e798acdefedfb3be1ebbab086940de4fe8ebabe60f1de34642004b7d`。适配CP、after75批/1500例原件及812240单次收费、Q012签字、Q013四路公式/分母/600项finite/1500行为配对及adapt75全套完整解码均通过；录像18118帧/36049控制样本，75工程操作closed，终止帧准确。

固定DIAG成功由382增至1051/1500，失败1113至449，超时5至0。正式GPU同一before状态9366条上的动作+0.1值由0.0192690352309至0.00471890637915；同一after状态15110条由0.0737464808282至0.00226361836703。两套状态分布不同，不能互相当同分布对照；正式恢复仍须A/B/final。

同原driver已按既定工程条件进入S1.2，当前记录9次main完整迭代、总822次/20201472学习样本，最近20Adam及loss/梯度finite。S1.2仍N512/rollout48/5epoch×4逻辑minibatch/micro512/LR3e-5/clip1；teacher控制回合10%、Actor/Critic44参数可学习，std与全部normalizer冻结，沿用原Adam动量；主段上限25634完整迭代/629981184学习样本，既定门禁/恢复条件不变。入口和首main周期CP另作纯CPU独审，不暂停现运行。S1_RESTORED及S2/S3尚未取得或启动。


S1.2入口和首main checkpoint独审通过，见`docs/v09_s1_2_entry_independent_review_20261120.md` SHA `0e700ec2be8b086288767e23ad7bf52ae69bf343e4588f364871b2a7eadfd457`；本机proof SHA `dc6eb862f823759a6a6f6fed821df8407e582d0dd4759c7c6f426841ab29a4a0`。入口53模型state/完整Adam/RNG同adapt；首main旧18各16280、新26各20，std无state、std+8normalizer同initial；账本5226及journal1646固定前缀证明after义务先闭合、Q009恢复后首轮24576单次计费及512新单调episode IDs。requires_grad/活Python ID/GPU恢复过程的序列化边界明确，不外推行为恢复或后续全部训练。

只读监督已建立（/tmp观察器，不改运行/账本或读封存final），监测后续checkpoint、正式门禁、预算/工程异常、意外退出及阶段marker，出现关键事件即回报主对话。原driver持续训练，后续仍按既定固定病例和正式资格推进SDS。
