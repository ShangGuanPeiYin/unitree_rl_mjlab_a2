# A2 0.2正式协议运行记录

依据 `doc/sds/SDS_A2_IdealHeightMap_Stairs40_v0.2_Formal.md` 修订1.2；目标代码0.2.1/schema3，协议v02-formal-climb-v1。用户已授权实施、独立审核、资源探测、C基线及一次seed42正式训练。Fix3已完成独立复核、真实512资源及在线验收；正式R0持续按原预算训练，最新结果见下方追加记录。下文历史待执行记录保留，以最新追加状态为准。

## 协议与训练流程

固定C源为 `logs/a2_ideal_stairs_v02/local_sds_20261111_1621/comparisons/C/chunk_0010_n512/checkpoints/budget_exhausted.pt`，SHA256 `7d94810bb86e6997fe2a0f03883a6a0e4ee594140bb3ff65deb86642c3938d9a`。迁移完整Actor编码器/融合/归一化与std，新建Critic、其统计及Adam；旧实验不是本协议resume。旧0.2任务与报告保留。

新训练以安全动态到达平台作为+10终端成功，停车独立评估；每脚独立同平台支撑面连续2个控制采样，记录最近1秒历史，同时当前四脚安全投影、当前至少两脚支撑、机身越平台0.8m/侧偏≤.2m、无失败连续5周期。真实失败优先，成功只给一次；timeout truncation正确bootstrap。停车在首次攀爬成功后归零命令，最多追加10s，四足真实安全支撑/机身侧偏/速度<.05连续150周期，不继续要求旧停车线；保留攀爬记录。

保留C落足每脚每级+.25、高水位进展、防重复、失败帧抑制、连续奖励数学及权重；std直接复制C并限[.05,.30]，entropy .005。物理200Hz、控制50Hz、地图10Hz、动作缩放.25；rollout48、epochs5、minibatches4；初始lr1e-4、[1e-5,3e-4]、目标KL.01、clip.2、gamma.99、GAE.95、value系数1、梯度范数1、clipped value开启。

R0/S1/S2/S3/S4/S5上限3000/3000/4000/4000/5000/7000，总26000更新；N/h分别1/.08、4/.08、8/.12、12/.16、20/.20、40/.20。每100保存、阶段局部每200联合验证楼梯三速各20+平地站立/三速各20×30s；同一checkpoint联合通过连续2次晋级。楼梯各≥18/20，平地无真实失败、站立速度模长<.05、行走平均vx误差≤20%。R0另有规定轻扰动各20门槛，失败清连续计数且继续消费3000预算。预算末次先验收到期门槛，未过即停；预算不转移、不追加种子。

R0槽位40%平地行走+10%站立+50%当前阶；后续25%/5%/20%上一阶/50%当前阶；最大余数分配，258边界纳入测试。S5仅使用规定400更新出生扰动ramp。最终固定S5触发模型做独立三速度各100、各≥95并平地验收，测试不选模。

## 监控与恢复

唯一GPU队列串行训练、评估和录像；资源档严格1024→512→258，首个完整通过即选。NVML整卡≤7168MiB、进程≤min(5120,7168−启动时其他占用)、host available≥4096MiB。只因实际资源故障降档，保留账本、新回合恢复；不能因学习失败降档。

W&B复用用户现有登录；SDK0.31.0公有云API为https://api.forge.coreweave.com。实际已认证username xingshenxu、entity xingshenxu-。执行器需复用用户终端既有 HTTP_PROXY=http://127.0.0.1:7897、HTTPS_PROXY=http://127.0.0.1:7897，未读取/记录key。project若无既有设置使用a2-ideal-stairs-v02-formal。正式run仍须实际上传并远端读回初始化记录、核对面板之后才执行正式更新。

监督器为唯一写入者，worker每更新本地持久化，在线每10更新或30s先到汇总、评估即时上传、等待时30s心跳。保持同一run ID、事件账本、已发游标、global_iteration与实际env_steps连续。120s上传失败要求worker更新边界保存暂停monitoring_blocked，不能默默离线训练。诊断在独立进程，恢复先完成新环境/reset再恢复全部训练RNG，下一段随机序列不受诊断次数影响。

## 当前真实证据（持续追加）

- W0：implementation/baseline.json 保存原dirty状态和源码SHA，未清理用户修改。
- 环境14 CPU反例通过，implementation/environment_report.md；独立审核进行中。
- implementation/migration_actual_1000.json：七场景真实1000观测，Actor均值误差0、std完全一致、Critic统计fresh，0PPO更新。
- implementation/gpu_ppo_smoke/result.json：2环境完整48×5×4更新，43个Adam参数状态、48诊断、指标有限；真实reset后全部训练RNG恢复并与独立诊断过程隔离。此为96样本工程检查，不是正式训练，也不是1024/512/258资源选档；2环境严格schema3恢复未执行，保留检查不绕过。
- 资源全生命周期、正式C基线、W&B正式在线run、R0和后续课程待实施审核闭环后执行。

实际运行命令、PID、W&B URL、资源档、阶段成绩和最终checkpoint SHA将在真实执行后追加，不提供尚未验证命令冒充完成接口。

## 2026-11-12 首轮真实执行与监控修订

原run为 `logs/a2_ideal_stairs_v02_formal/local_sds_20261112_135450`。独立Sol/high审核关闭初轮环境/模型/编排问题；完整CPU148项通过。1024进程显存观测峰值5196MiB超过5120MiB守卫，退出并释放CUDA后512通过55更新+S5评估60例+新进程恢复1更新；512更新计时median3.46448s/P95 3.58481s，吞吐7075.29环境步/s，整卡观测峰值4515.5MiB、本PID3172MiB、host available最低15652.39MiB。详细原始关闭采样与证据边界见 `logs/a2_ideal_stairs_v02_formal/implementation/resource_review.md`。累计探测57更新/1425408样本，正式预算独立且仍为0。

实际W&B run为 https://forge.coreweave.com/wandb/xingshenxu-/a2-ideal-stairs-v02-formal/runs/37b559428d704af09bf16d7ed646595d 。已通过远端API读回初始化和评估数值，并从远端保存Report读回8组面板定义及准确run过滤；无可连接浏览器，未声称视觉核对网页渲染。26个已核对远端事件与本地数值无差异，详见implementation/online_checks。

C新契约基线三速各20/20攀爬；平地站立mean_speed=.00284752m/s无失败，.2/.4/.6m/s的平均vx相对误差31.5356%/23.0261%/21.3646%，高速20次真实失败。基线不合格不阻止R0，但不能称恢复成功。

首次正式更新前发现140个评估回合被排入后续训练聚合队列，独立审核CPU反例确认O1；尚无train事件被发出。已边界暂停，supervisor和worker均退出、CUDA释放。暂停checkpoint为原run/formal_seed42/chunk_0000_n512/checkpoints/monitoring_blocked.pt，SHA256 `17097a7b98af8413378a3c9bd42cfc3b89a8ddfb07eda393a893375d8de9b65e`。其完整模型/Critic/Adam/RNG/课程与正式初始化逐树一致，learning_state=false/0/0、所有正式消费0。

修复由原编排实施者进行：只允许训练worker记录进入训练聚合，诊断污染写隔离证据；补全本地面板身份元数据。显式零预算监控修订入口将在新目录保留同一W&B身份、全部学习状态和旧审查链；普通resume仍严格拒绝源码/契约不兼容。独立复核后以已选512完成同档完整生命周期复验、重录新SHA基线，再开始R0。不升回1024，不把探测模型作为正式起点，不重置或扩大26000预算。当前尚无正式学习结果。

## 2026-11-12 真实50更新与Fix3监控修复状态

`monitor_restart_20261112_145718` 已通过O1/M1独立复核（新approval_online_fix.json）、164项全项目CPU、新source512完整55更新+S5评估60+新进程resume1。新探测56更新1376256样本；与父探测57/1425408分开，累计113/2801664，不混正式。新正式header SHA `bb880a56cfaf9ba1f73dea33a1a4e2a6cdef185170bd1effc2b93360c1650a89` 的八学习/随机状态与父逐树一致，Adam空/预算0，无探测污染。新归档SHA `c70cfd0c88e716e9b6893071d52aa850463e37a1f617963c8a52a2d13234769c`。

同W&B ID的源码配置修订、初始化和8面板真实读回通过。新C基线三速度各20/20；平地80例完成，.2/.4/.6行走误差31.5427%/22.9987%/21.3797%，高速仍20真实失败。此为实际结果而非效果改善。随后R0实际训练50更新，512环境、每更新24576样本，共1228800环境步；learning50/1000、43个Adam参数完整。尚未到第200更新联合验证，不能称R0通过。

首训练窗口在200块完成前已远端可见：第10更新/245760样本/504训练回合，分母/回报/成功率与source一致。之后在线积压超过120秒，worker按控制请求在第50完整更新保存退出0，supervisor退出1，CUDA释放。暂停checkpoint为该run/formal_seed42/chunk_0000_n512/checkpoints/monitoring_blocked.pt，SHA256 `963ffb433b0c2c62ec4c85498ba23e8cca864ec89b7dac2cdb898ccd4ef06601`。不退款、不因监控问题降环境数。

受控同ID补传385.53秒成功，88/88事件真实确认，supervisor预算/checkpoint文件字节未改。uncached远端复核88事件11504数值与全部28训练窗口（50paidupdates/3441episodes）一致，无诊断混入、重复或丢失；证据implementation/online_checks/recovered_0050_remote.json。

独立审查另证实C1：旧heartbeat77/86及resource_close87把真实40/50进度误报为0，真实远端同样存在。C2为本地聚合使用整队列ACK时钟，网络积压时50更新被分成28窗口，放大队列。旧历史错误保留，不删除/回写事件。已交原实施者Fix3：独立本地聚合时钟、持久最新真实worker进度、同周期训练窗口/心跳去冗余，保留120秒及uncertain屏障；专用监控修订迁移须保存全部50更新Actor/Critic/Adam/RNG/课程/账本和同W&B身份，不能使用只允许0预算的迁移入口。待独立复核及新512完整复验后从50继续，下一联合gate仍200，不把50模型当C基线、不从C重新训练。当前没有第200评估、R0晋级、停车或40级结果。

## 2026-11-12 17:13 实际进度：R0 600，继续原预算

Fix3独立68项、root完整175项CPU通过；新源码归档SHA `3f84349d4f639b320b5fc57c376cb0a6bdaed334bd9da68fe7ebdad0dc423f73`。实际新run `monitor_clock_restart_20261112_160539`，supervisor PID1960593，session38768/只读monitor36404。50学习态header SHA `42d62bf1fe88f09496c240a298aa878921add85214257a46522bc651b97abac3` 与父8树/全部消费账本精确相同，普通正式worker从50继续。第三轮512完整55+S5eval60+resume1独立PASS；累计资源探测169更新4177920样本，正式单列。没有重C baseline或重置Critic/Adam。

真实GraphQL精确分页117/117事件、19444数值/全ACK ID完整覆盖/无重复、source revision89后无横轴回退。实际200 force/资源收尾/checkpoint全部为200/4915200，schema3为200PPO/4000Adam。SDK扫描旧88各双份的原失败保留，不隐藏去重或假定根因；独立GraphQL表示及全部本地发送意图无重复。只读约250秒9新事件测量：GraphQL首次可见延迟观察上界中位9.47s/最大19.61s，SDK扫描14.92s/26.79s；请求中位1.91s/9.23s。含5秒轮询及串行查询开销，非精确传播延迟/根因证明；观察队列最大2、最老事件年龄8.90s，保持源码/120秒规则不变。

R0第200联合FAIL：攀爬三速各20/20；平地站立norm .055890不合格，.2误差248.6693%，.4误差.5924%通过，.6误差61.6893%且20bounds失败。第400联合仍FAIL：攀爬各20/20，站立norm .008267通过；.2误差94.6389%、.4误差76.7748%、.6误差70.5904%且20bounds失败。没有因暂未晋级调整配方。

实际已到第600更新/14745600环境步，checkpoint SHA `2d415988f72ba12d05557dd28841791f6828a4be76f9f7331e4bf603523190cf`；600攀爬三档又各20/20，平地仍评估中。R0连续通过仍0，预算上限3000、剩余2400；未进入额外扰动或S1。完整逐速度/模型与case SHA见implementation/training_gate_progress.json，实际原始summary在各validation/flat目录。工程和监控通过不等于平地恢复或40级通过。

600联合验证完成，仍FAIL：攀爬三档各20/20，站立mean_speed=.006061通过；.2误差31.2136%，.4误差16.7946%但8次真实失败，.6误差57.4841%且20次真实失败。继续R0向800，streak0，上限3000不变。独立learning_validation_diagnostic.md对200/400确认命令/坐标/30秒/bounds/案例/比例/Actor无特权泄漏，未发现新工程阻断；学习未达标不触发改配方/降档。

R0第800联合仍FAIL：攀爬各20/20；平地站立norm .006118及.2速度误差5.1675%且无失败通过，.4误差16.4579%但18真实失败，.6误差21.5941%且20真实失败。继续1000，streak0，上限3000/剩余2200。800 checkpoint SHA `6b923052bd347417a8a3b8c4df9500dcedd6f0117eb3bef24fc967a5c8ed04cc`；累计正式19660800步。r0_0800_remote.json257唯一/51269数值通过，资源峰值6372.75/3172MiB及host最低11788MiB均在守卫内。

R0第1000联合仍FAIL：攀爬各20/20；平地站立norm .005942及.4误差13.2530%无失败通过，.2误差29.5074%，.6速度误差17.5191%但20次真实失败。继续1200，streak0，上限3000/剩余2000。1000 checkpoint SHA `ad7eadb1d283b415ce7dace2419d3631769d957ad4724d88338a7d526d78f6d2`，正式24576000步；r0_1000_remote.json308唯一/62845数值通过、无轴回退。

R0第1200联合仍FAIL：攀爬各20/20；平地站立norm .004168、.2误差3.9066%、.4误差2.3859%均无失败通过；.6误差39.7477%且8次真实失败（5bounds、3nonfoot_persistent）。继续1400，streak0，上限3000/剩余1800；1200 checkpoint SHA `2ec1d438839841999d7dc058277b327548d9d53f7efc57dd1af187ca88ad2730`，正式29491200步。r0_1200_remote.json356唯一/74128数值验证通过，无新横轴回退。不能拼接不同模型的通过档位宣称联合通过。

监控事件补记：18:21:59曾因确认超时触发120秒保护，此前1200训练worker已在18:20:16退出并释放CUDA，完整1200模型已保存；期间仅独立评估，无PPO更新。上传确认恢复后下一训练worker在18:31:49启动，未清预算/未绕过120秒。supervisor.status保留历史monitoring_blocked标记，当前telemetry阻断标志已清；具体上传/可见性/查询延迟根因未确认。证据online_checks/diagnostic_guard_r0_1200.json，保持原source继续，不为猜测改代码。

R0第1400联合仍FAIL：攀爬各20/20；平地站立norm .005707、.2误差18.5993%、.4误差7.2805%均无失败通过；.6误差32.3219%且20真实失败（15bounds、5nonfoot_persistent）。继续1600，streak0，3000上限剩余1600；1400 checkpoint SHA `96fab6f5371d720423a6a8aa739dba4ba1c0da84b3313d750d6ca2ea2f83c10d`，正式34406400步。r0_1400_remote.json407唯一/84596数值通过，源与配方未改。

R0第1600首次联合PASS：攀爬各20/20；平地四档均20回合无真实失败，站立norm .006276，.2/.4/.6速度误差6.2852%/8.5150%/14.3857%。实际下一worker budget记录1600 joint=true、连续通过1。继续1800，仍需第二次联合通过及seed300142规定额外扰动才晋级，不能用本次固定初态重复当独立随机工况或40级验收。1600模型SHA `36b45df09b2420d04885062f0b577de9fd2ba9cf14efa88ac366ecdf168915d2`，正式39321600步；r0_1600_remote.json457唯一/94252数值通过。

## 2026-11-12T19:58:46.596097+08:00 — R0 1800联合失败与零更新暂停操作恢复

实际1800更新/44236800正式样本：三速度攀爬各20/20，flat四档各20无真实失败；站立3D速度0.005820m/s、.2/.4/.6误差19.462/6.297/28.394%，最后一档超20%故jointFAIL、streak1→0，不触发扰动或晋级。root预置control reason不符合原接口导致chunk0009在initialized后首PPO前报Unknown worker control request；保留原失败control/result/log。使用原worker正确reason monitoring_blocked从该initialized恢复至新engineering_pause_recovery目录，真实保存monitoring_blocked.pt，8棵学习/课程/RNG树完全相同、1800/36000 Adam计数不变、无updates或inflight。备份原supervisor_state后用真实result收录零新增预算恢复点并清除已消费1800gate输入、chunk指向10，避免重复消费；未改生产源/配方/预算/环境数。证据pause_1800_recovery_launch.json及pause_1800_recovery_evidence.json；W&B余2事件正以原发送器确认，M2候选尚待独立审核/未安装。后续课程及40级能力尚未验证。

## 2026-11-12T20:08:52.354819+08:00 — M2独立审核闭环与1800全学习状态正式续训

新增独立Sol/high review_curriculum_counter_fix.md及approval_curriculum_counter_fix.json（审批SHAe564b52d4894f40192cc5932414549184896828e098384d427b082f19d3a92f6），M2已关闭、允许绑定helper复用真实已闭合512生命周期。实际CPU helper准备新run /home/xxs/dog/unitree_rl_mjlab_a2/logs/a2_ideal_stairs_v02_formal/monitor_counter_restart_20261112_200716，learned header SHA5065a9f16c08fa2b888a32fdb6d20424a50d4decb44eccb4fb27a1bf5a1a69ee；root独立核对8树、预算账本及531历史事件字节均相同，父state/checkpoint/events/telemetry冻结SHA不变，证据learned_curriculum_counter_migration_actual.json。明确未新增GPU probe，不写新probe passed/completed；原生命周期在parent历史，原运行时守卫仍执行。新supervisor PID2092734、命令/路径/日志保存在launch_curriculum_counter_fix.json，--resume-run同ID37b559428d704af09bf16d7ed646595d/seed42/512/online，从1800继续，下一gate2000，保留R0剩余1200。实际新source在线读回和worker首完整PPO正等待核验，未宣称学习通过或40级验收完成。

## 2026-11-12T20:17:16.232946+08:00 — M2实际在线独立验收闭环与1900保存验证

新增独立Sol/high review_curriculum_counter_live.md（SHA6c317f37597d156b9429ba0f456e359a814b6d9266e741e5bde3b02119a8078d），M2实际上线验收完成：父pause→新header→worker初anchor的8树不变、531历史前缀保留；新source_revision基点1800，实际首1801开始无重置预算。GraphQL首窗event534/global1809的全部课程字符串/数值与本地逐项一致；完整534唯一/104527数字匹配；独立本地534–542九窗raw来源/分槽统计/预算SHA复算一致、30.125–30.739秒生成，不等块结束。远端验收范围到1809，本地范围至1878，后续544暂未确认快照保留；root稍后实际545及547全部ACK/无barrier，不称网络永久无故障。root新增curriculum_counter_first_worker_anchor.json、实际在线字段报告及curriculum_counter_real_1900_checkpoint.json：1900真实正式更新保存SHA c5764e033e1c191f6bb7360a4f9555544bfe6657e9c272c246e70ad577cd41e2，schema3/完整43Adam/38000步、46694400环境步/512通过。无新GPU probe、无配方/课程/预算变化；继续R0，尚未完成2000联合评估或40级验收。

## 2026-11-12T20:34:38.526691+08:00 — R0 2000联合验收失败并继续原预算

更新training_gate_progress.json/运行文档/progress及training_statistics_partial.json：2000模型SHA20171e066610bef860e136f453ce1383d643eeaacacbdb58e5a870a88a5dd785，三速度climb各20/20；flat四档各20无真失败，站立norm0.002645m/s、.2/.4/.6 body vx误差29.4858/6.9972/25.6021%，jointFAIL/streak0、R0边界剩余1000，下一2200。全部raw140唯一案例/模型案例SHA/统计冻结/逐档速度均值已只读复算匹配；固定案例不称独立随机工况。1–2000正式更新/49152000样本连续无重复，probe未混入。2000训练块资源observed card5629.75/PID3172MiB/host最低13834.46MiB通过，未新增GPU probe。监督器已继续下一块、不调参/降档/退款或扩预算；40级能力尚未验证。

## 2026-11-12T20:57:52.799457+08:00 — R0 2200联合验收失败并继续原预算

更新training_gate_progress.json/运行文档/progress及training_statistics_partial.json：2200模型SHA68378747f019b5f790783b986547bbd1a2313120aa538a5ba873e16f57d6ee82，climb三速各20/20；flat四档各20无真失败、站立norm0.002474m/s，.2/.4/.6机体vx误差21.5899/3.6047/6.9331%，仅.2超20%故jointFAIL/streak0。raw140案例唯一/模型案例SHA/统计冻结/速度均值复算相符。正式1–2200连续54067200样本，probe未混入；R0边界剩800，下一2400，不因接近阈值放宽或调参。2200块观测card5770.75/PID3172MiB/host最少13956.62MiB通过；未新增probe，后续stage/final仍未验证。

## 2026-11-12T21:22:06.365171+08:00 — R0 2400联合失败与原预算继续

更新training_gate_progress.json/运行文档/progress与training_statistics_partial.json：2400模型SHA0af62be2be4372da0486022b5351a1003256dd51db05c9c0c5dc83c69243a392，climb各20/20；flat80均30秒无真失败，站立norm0.001648m/s，.2/.4/.6机体vx误差47.9771/6.2028/10.5829%，仅.2失败故jointFAIL/streak0、R0剩600，下一2600。140raw唯一案例/模型案例SHA/冻结统计/逐档速度均值复算匹配；正式1–2400连续58982400样本，probe独立。2400块observed card5764.375/PID3172MiB/host最少13630.85MiB通过；无配方/环境数/预算变化，低速仍未恢复，不宣称学习改善。

## 2026-11-12T21:44:30.217109+08:00 — R0 2600联合通过一次，继续2800

更新training_gate_progress.json/运行文档/progress与training_statistics_partial.json：2600模型SHA69267bd5702be3bea27a3bb3fd25cca4aeb4a868a12a7f675d90ff57a4391ab9，climb三速各20/20；flat80均30秒无真失败，站立norm0.001579m/s、.2/.4/.6机体vx误差4.7212/4.8420/18.6108%全通过，jointPASS。实际新worker已消费2600门槛、streak1/r0extraFalse，仍R0；需2800再次joint及R0附加60扰动才晋级，不继承1600已中断的PASS。140raw唯一案例/身份/统计冻结/逐档均值复算相符。正式1–2600连续63897600样本，probe独立；R0边界余400，未调参/扩预算。2600块observed card5864.44/PID3170MiB/host最少13315.45MiB通过。尚未宣称R0恢复或40级通过。

## 2026-11-12T22:09:02.559652+08:00 — R0 2800联合失败并按原上限完成末块

更新training_gate_progress.json/运行文档/progress与training_statistics_partial.json：2800模型SHA71bfeb61e92ab20ca462dd67357a4a385bb8693387ba5c2474de7d9f0bee3991，climb各20/20；flat站立/.2/.4各20无真失败通过，standnorm0.002099m/s，.2/.4/.6机体vx误差7.2113/3.1310/15.5310%。.6虽速度误差过，但20/20均bounds真失败（27.64秒、max|y|4.00404m、max heading0.42731rad、该档27640控制采样），故jointFAIL/真实新worker streak1→0，不忽略真实失败。固定初态重复不称随机独立工况。140raw唯一案例/身份/统计冻结/速度均值相符；1–2800正式连续68812800样本，probe独立。R0剩200，按原3000上限继续末次训练与验证，不提前跳阶段、不加预算、不改终止或侧偏定义。2800块observed card4601.25/PID3172MiB/host最低16172.68MiB通过；无R0附加扰动/停车/晋级发生。

## 2026-11-12T22:31:20.963811+08:00 — R0 3000末次联合失败，真实预算耗尽终态

更新training_gate_progress.json/运行文档/progress、training_statistics_partial.json并新增final_training_statistics.json：正式1–3000严格连续73728000样本（512环境），probe169/4177920独立；实际末次eval模型SHA4640cf0b481229773e07400c141f2904a582dcfddba740cbe2e02709b28d82cd，climb三速各20/20，flat80均30秒无真失败，standnorm0.002237m/s，.2/.4/.6机体vx误差64.1940/0.2516/16.6474%，仅.2失败，jointFAIL/streak0。140raw案例唯一/模型案例SHA/统计冻结/速度均值复算相符。零新增PPO门槛消费后实际final.json status=budget_exhausted/R03000其余0，终态完整checkpoint SHA2ab41ba8cdf18cf1079311af53f50d111e537546030cdf2ccaebcb333f73ff81（仅追加末次gate状态，模型同策略的树等值仍待独立复核）；停止新增训练，S1–S5/扰动/停车/finaltest未触发。代表三速录像/W&B终态收尾正在执行，尚未宣称最终交付闭环。3000块observed card4528.5625/PID3170MiB/host最低15988.92MiB通过。新增final_training_audit.md预算部分已独立确认30保存点/17anchor/43Adam60000/144000控制记录及25251000有限float，末次gate/终态/视频/远端F部分仍待实据。未扩预算或改配方。
