# A2 v0.7 独立最终工程验收

**结论：accepted=true，blocking=0。** 签署时间：2026-11-17T15:49:52.192633+00:00。本结论准入从精确stop_004817.pt继续唯一seed42的v0.7训练，仅适用于下列冻结闭包；不是训练目标完成、不是最终300例通过，也不允许另选模型、fresh或变更配方。

reviewer为/root/c1_reviewer，按主协调显式gpt-6.1-sol/high派工；此为所提交配置，不另声称底层运行自检。reviewer没有实现产品或执行者tests，只写独立审计文档和CPU只读核验脚本；无GPU、PPO、物理仿真、Git写或生产启动。已有用户完整实施/验证/自动续训授权，曾有授权视图阻塞已由用户直接授权解除；本签署不以发布授权替代测试。

## 固定输入与审查覆盖

|输入|实际SHA256|
|---|---|
|SDS|`303ae40a5c5e3fe82f9f0df3aae8d153136e016439087c8db9701aeb17197aca`|
|唯一源stop_004817.pt|`a41963a8bf5e6f1d4810b0a0c912ac65747d61bd481ea3432ef638a99e076e64`|
|v2 source closure|`829683d61d7732cf683c96ebc6f659769e60c7324a386ba95227ac3fb53e274a`|
|v2 contract|`b8857d4b18482f391eb9ec713ea8b6a7236af77bb14d7f8fdd6bf9f931fbba77`|
|freeze/v2/V07_SOURCE_FREEZE.json|`f9f67f08a7120b479ed6dfe3dd9916df2c8492ee1e6971d72f807da60f890aa1`|
|完整whole-branch源包|`c7688058064322790939e4398c9cfa52593202b6ac2ab3ff17977f3b34b5487d`|
|ENGINEERING_SUMMARY_v2.json|`2a13a9f29a8921aed1579585c748402f8a54263c83f73e7ea5f9de9c788c3ebe`|
|不可变工程报告快照|`519de6e85b9ee23722734991c8a19c7c51f3c1d224d9d0753d39d823f5544f40`|

94源码、22测试/支持文件、30文档、3根observer输入均逐文件实核，形成机器verdict完整map；旧97源、28tests、8项preserve原件也实核无漂移。完整源包10951行/57targets按真实冻结C1 baseline逐hunk重建全部匹配；57/57目标均归入此前完整读过并闭环的模块scope及明确metadata delta，62共享依赖与旧before相同，没有另留未审的变更代码。未重复无变化套件或将namespace归一视图当真实baseline。

Qv07-001恢复裁决SHA1f84b142b70cc952eeca6e0a630f76d5d28f3d3b542cf2bcf5084d19be94a3cf、Qv07-002控制裁决SHAdad4580188773b0345f3f741b5f8a3d9f988652d7a257c8e2205dcd1d23adbd1均全文已读并纳入合同/文档绑定；SDS原字节未改。PROGRESS.md所有裁决/延后项及当前WORK_REPORT.md已读，旧执行中段落由其已取得完整证据附尾更新；本签署不把这些持续文件或codex_changes封成不可变输入。签署时正式目录仍不存在。

独立最终复算：[输入与累计审查覆盖](INDEPENDENT_INPUTS_VERIFIED.json)，SHA `1f14281b5af9b2e3985eb1520e85f861bb007fbb9346da64c59ea33f0f2eac70`；[原始工程证据复算](INDEPENDENT_EVIDENCE_VERIFIED.json)，SHA `a89b2853841a3539f0bddd1ce4415eaad0c471d2bada2f249168e06ab90b2f5c`。前者映射完整57targets和149个冻结输入；后者复算197项实际原件，不依赖ROOT_AUDIT摘要得出主要结论。所有原件保持不动。

## requirements验收结果

|要求|核对结果|
|---|---|
|D22及原地形保持|A1几何/native表已审；原0..21几何与原点保持，新增物理22、6级/.20高/.22深/.80宽/.01板/2m平台，物理23而最终L40仍21；实际8env D22 smoke两更新通过|
|独立课程顺序|按(1..17,22,18,19,20,21)派near/easy，禁止物理ID直接代表难度；CPU及真实factory路由覆盖D22/20和flat0/H001|
|新D/L退级|训练64窗EMA只诊断，H规则保持；当代gu<1000不性能退级且不挡正常晋级；完整当前60例、三速各20、S≤12 rolling3、相邻100、delta<6才退/最低停，已审并CPU验证|
|保持与最终门槛|保持各15、≥12失败告警/独立两次停止；原最终三速各100且各≥95保持。工程基线不进入正式资格/streak，不增加成功率准入|
|一次迁移/重复恢复|唯一源SHA+完整state树校验，首次gen72→73/gu0且首4818；二次/native不重迁移、不清gu/queue/保持时钟；普通终态/partial/工程容器拒绝正式resume|
|模型/Adam/元数据|真实原source逐块对拍model/std/full45 Adam参数ID与完整groups/step/momentum、normalizer/learning/transfer metadata；当前lr保留7.593750000000002e-05|
|Q010与消费事务|首terminal或精确F0 500窗、raw/hash/summary/sidecar、当前validation直接handoff；prepared/applied及64行消费proof、IO异常恢复不双PPO/EMA/评估/退级；两项核心P2已R1闭环|
|外部control|真实runtime文件接线，continue/pause/stop、首次有效锁存、无锁存坏JSON工程恢复、未解决有效锁存早拒绝、原字节留存、不覆failed、终态不重写、更新中当前完整计费但下一PPO不开始均核过|
|配方/旧版本|奖励/PPO/网络/观测/出生/探索/物理终止保持，真实旧/新数值parity；原旧代码/测试/源checkpoint保持字节，native导入隔离|
|预算/CLI/日志一致|D25/D22/D20=6000/6000/9000，D21000/global48000/formal48100；same合同供budget/log/CLI/recovery/supervisor；observer只读且V/R独立stage统计已修复|

数学D22平台起点1.32、机身线2.12。实际float32 N*d=1.3199999332427979，bodyline=2.119999885559082，与float32字面2.12一致；strict>=及原终止epsilon没有改。数学边界单元采用float64避免双舍入夹具误断言；没有把它冒充GPU已登顶证据。

## 同v2完整CPU与累计审查

native最终完整CPU：**236 collected/236 passed，0failed/0skipped/0errors，exit0**；原文SHA5ac3207a1388681967e191760c4fb649a44f24658f2622848236a367ac462313。旧版完整兼容：**587/587 passed，0failed/0skipped/0errors，exit0**；原文SHA52189e6f5f91c01f1e2073947faf6894526958b8326c5abbeff2d507a665d894。命令、env、原文summary/exit、结果JSON及输入不变均独立核对；旧兼容不冒称native闭包测试。v1的219项虽通过仍被独立反例挡下，修复后另冻v2并完整重验，不沿用旧闭包准入。

累计五项确认P2均已闭环：policy缓存动作/receipt身份、P50证据来源/首终态、observer None阶段，以及B核心缺窗口消费proof、控制锁存跨异常恢复两项。原RED/失败更正/独立反例均保留，当前无确认未处理blocking或被静默推迟的minor。本次没有新建产品tests或重复已过同SHA套件。

|累计独立报告|实际SHA256|
|---|---|
|[REVIEW_PREFLIGHT.md](../coordination/REVIEW_PREFLIGHT.md)|`db1d14f0a3010506c916b026dd84fa807dcb2bfd15eb5728e1b232fdfa59c1ba`|
|[TASK_A_REVIEW.md](../coordination/TASK_A_REVIEW.md)|`95e63571542276b57c5ba90696ef3a492574bf4ae7f7ea05bcd1bb43512d015b`|
|[TASK_B03_REVIEW.md](../coordination/TASK_B03_REVIEW.md)|`471c6777cf9b52345ff364131fdf94a617d25470e7bd02a8c1bda44ff62c63cf`|
|[TASK_A2_REVIEW.md](../coordination/TASK_A2_REVIEW.md)|`8e720f6250a3932af11a6ba7ad52520bd5b2577315ff2d16a9a87a568d023815`|
|[TASK_ROOT_REVIEW.md](../coordination/TASK_ROOT_REVIEW.md)|`3d752e68c9ae8861b7e5abe0e9ad956ee3d399361ce1706adaa7d624b73e999c`|
|[TASK_B_CORE_REVIEW.md](../coordination/TASK_B_CORE_REVIEW.md)|`3fbd00fcc66a409be7a8b886452333b37dd43bb582f6040d4bfacb94bd84b5a4`|

## 五GPU实际工程证据

五项均为同v2 source/contract/freeze，实际command/env/exit0、stdoutSHA、engineering_manifest已核，输入changed_files=[]、生产目录未创建。按started_at+duration核严格单worker串行，无重跑挑种子。

|阶段|实际证据与费用|
|---|---|
|smoke|seed42、真实8槽全tile22，2 PPO；768control/3072physics/40Adam|
|probe|真实512训练、评估容量8，R/L40/tile21，5warm+50measure+1recovery=56；1376256control/5505024physics/1120Adam|
|source-reload|原4817两次实际512裸load，0PPO，首env/model弱引用释放后才构造第二次，loaded/ready均4817|
|baseline|原4817零学习45例：D25、D22、D20×(.3,.5,.6)×5；各首回合原件、模型前后全state digest不变|
|retention-protocol|原4817零学习flat/H00各15，flat15项均非终态真实500步window_end；H00首terminal原件完整|

probe50独立现场capture与真实boundary_000050.pt再次交叉核对：update50、Adam1000、model rawSHA43b8bb61c4d5cd8a4cd0b90cb789f708112d6f507de9a6c560c2ce12c4f928a2、完整Adam digestbf94ddc349f5b02d7bb9160b7e956a2676cf0a18ef314bc2e661235acf2ed3f4一致。15个实际first records独立重新restore/三SHA/主entry/物理case/slot/steps/outcome校验通过，没有使用payload自证身份或拿55模型替50。

55 checkpoint与resumed55 loaded的model/Adam/normalizer/learning逐块相同；实际56后Adam1100→1120、控制+24576、物理+98304，训练commit1..55连续且恢复只commit56，无第二次ETH初始化。工程容器不能供正式resume。

双512 load的四个实际loaded/ready checkpoint分别与原source model/full Adam/normalizer/learning/transfer metadata逐树对拍；schema9/gen73/gu0/D25、phase_iteration2117、角色102/164/123/123、frontier17/near15,16/easy1..14、无训练tile_override均符合。逐槽512 role→actualtile核过，全部28缓存检查true。第一reset181+512=693槽次，第二164+512=676槽次；extra181/164明确属于Q001允许的额外初始化消耗，临时reset不计PPO/课程窗/真实episode。四RNG时点中after_restore与实际输入checkpoint六RNG逐摘要相同，after_final与各ready checkpoint匹配；schedule/新episode会推进课程sampler与env/sampler，未seed/倒回/拼result，不声称位精确或物理连续。

## 起点行为、费用与资源的真实范围

|原4817模型|.3m/s|.5m/s|.6m/s|合计|
|---|---:|---:|---:|---:|
|D25|5/5|5/5|5/5|15/15|
|D22|5/5|5/5|5/5|15/15|
|D20|0/5|3/5|1/5|4/15|
|flat保持协议|5/5|5/5|5/5|15个500窗|
|H00保持协议|0/5|0/5|5/5|5/15|

这些是原模型起点，不是新训练改进；D20和H00失败如实保留，未增行为准入/重评挑结果。H00共10失败低于原≥12告警线，仅说明未触该线，不能说15/15成功；工程保持不初始化正式streak。flat分组summary的reason沿旧fallback可显示timeout，但实际15条raw timeout=false/evidence_kind=window_end、500步，不能把该标签当终态超时或失败。每槽500净位移start/end及原flat阈值已独立重算。

|工程费用|有效首回合control|额外等待control|实际control|实际physics|
|---|---:|---:|---:|---:|
|probe50诊断15例|54805|0|54805|219220|
|baseline45例|12365|875|13240|52960|
|保持30例|20400|4330|24730|98920|
|评价合计90例|87570|5205|92775|371100|

工程PPO共58更新、1160Adam、1377024control/5508096physics；加评价实际成本，GPU总**1469799control/5879196physics**。额外物理20820不漏计；另发布CPU2env一步2control/8physics、0PPO独立记录，不并入GPU。上述均不计正式48000；初始化forward/sense与reset没有伪记成训练step。

已观测资源均符合原整卡≤7168MiB、进程≤5120MiB、主机≥4096MiB、磁盘≥10GiB。probe明确是**updates6..55的50测量点**，三次diagnostic资源点都在50；50窗整卡max4223MiB、进程max3230MiB、host min16187MiB、disk min451.9088GiB。其他阶段及批次资源原件也已核。**55→56功能已验真，但其资源没有另采；不把50窗最大值外推恢复窗口或连续精确峰值。** 原v0.6§9.3和v0.7§6.2明示50测量+1恢复，并无第51采样准入线；据此不擅加门槛、不补GPU重跑、不改资源阈值。root也独立确认同一口径。

## 准入后的既定流程与限制

正式来源仍为4817/Adam96340/control118382592/physics473530368/lr7.593750000000002e-05，首次仅迁移gen72→73、gu0，下一更新必须4818，完成后应Adam96360/control118407168/physics473628672。flat/H00上次4800、下次5000、原streak0继承。源已消耗F0=200/H=2500/D25=2083/D20=34；账面剩43183、按未来phase cap最多32883，已完成阶段未用额度不挪用，不承诺跑满。

仍512训练/评估上限8、rollout48、5×4=20Adam、常规save100/gate100/diagnostic50/保持200。工程同速5例批实际5env沿原factory(num_envs=len(batch))，上限8未降低。D/L当前代100更新门禁资格及两次每速≥18/20正常晋级保持；1000保护不阻预算/保持/诊断/用户/资源/监控独立停止。奖励连续项按dt累计、0.4前进距离和终态+10/−10/超时0不改；PPOclip/value/entropy/KL/grad/std逻辑和原lr适应不改。

本结论允许root按已授权流程先无execute preflight（正式目录不创建/CUDA不初始化），再精确manifest/source/acceptance SHA校验后自动后台正式execute；reviewer不启动。签署时正式学习和最终三速各100且各≥95的300例测试尚未发生，不能报告成功或效果改善。Q008完整evaluation RNG隔离仍未建立；恢复是new episode，非连续物理轨迹。未来native段只读observer需新审计合同，当前observer只接受首4817段。普通终态/pause不能因本验收被重新授权恢复。

本次只新增独立验收MD/JSON及两份复算回执，没有修改产品、执行者tests、旧证据、immutable工程快照或Git。root统一在[logs/codex_changes.md](../../../codex_changes.md)追加本批审计与签署记录。原freeze中的accepted=false/core_review_pending是创建时状态，保留原字节；实际准入以本MD及机器verdict对同closure的签署为准，不回写旧freeze。
