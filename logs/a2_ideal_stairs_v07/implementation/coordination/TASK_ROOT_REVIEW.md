# ROOT五文件限定独立审查

> 最新限定结论（R1）：Spec compliance / Code quality 均 **PASS_WITHIN_ROOT_R1**；F-ROOT-01/P2已关闭，ROOT本包当前blocking=0。以下初审REQUEST_CHANGES保留为历史；本轮不是完整v0.7 accepted，原declined-to-judge范围继续有效。


日期：2026-11-17T14:03:08.723927+00:00；reviewer=/root/c1_reviewer，gpt-6.1-sol/high。依据ROOT_REVIEW_BRIEF、ROOT_REPORT、GLOBAL_CONSTRAINTS、完整SDS303ae40a5c5e3fe82f9f0df3aae8d153136e016439087c8db9701aeb17197aca及Qv07-001/002。完整读ROOT_REVIEW_PACKAGE.diff全部682行/38066字节，实际SHA `11a96f27335b8971aa9c2f361660d982fb1854ac6ae2c9d283b653cec48d8c3d`；五文件在独立反例前后均匹配ROOT_OWNER_SHA256。

## 双verdict与分级

- **Spec compliance：REQUEST_CHANGES。** 双load wrapper、显式预检shell和配方对拍的限定静态接口符合已授权设计；只读observer在正常可达V/R阶段存在确认P2，不能完成整个首4817续训段的持续阶段统计。
- **Code quality：REQUEST_CHANGES。** 确认P2=1（F-ROOT-01）；P0/P1=0，ROOT本包blocking=1。未实现修复、未修改产品或执行者tests；没有全局accepted，也未GPU/正式训练/Git写。

## F-ROOT-01 — P2：无substage的V/R阶段令observer退出且合并阶段计数

位置：`coordination/v07_observer.py:115`、`:123`，以及`:33`/`:109`的sort_keys JSON输出。observer直接令`stage=commit['substage']`并以stage作为stage_counts_segment键。

生产真实接口：runner.commit_learning_update写当时课程phase/substage；curriculum._gate_transitions在L末段完成后提交V、substage=None，之后V→R仍substage=None。因此首段此前必然已有D25等字符串键，正常进入V的首个已计费commit会加入None键。

实际JSON序列化会抛`TypeError: '<' not supported between instances of 'NoneType' and 'str'`，observing进程终止，后续更新、预算与保持独立解释停更。即使仅移除sort_keys，V/R仍共用同一null键、重读后还涉及None变字符串键，无法区分实际阶段；因此应修正稳定的阶段身份，而非只压制序列化异常。D22/D20原substage跨度应继续按真实物理课程和generation关闭。

### 已实际复现

用纯标准库和真实文件调用observe：第一次读取4818/D25/string stage成功并持久last_update4818；追加4819/V/None或4819/R/None且累计spent和update平衡，两种均实际抛上述TypeError，持久last_update仍4818。工具最终exit0表示成功捕获两缺陷，并不表示observer本身通过。

回执：`/tmp/a2_v07_root_observer_independent_counterexamples.json`，SHA `aa384b1931c7ab26fe28841795fa33aae7eeff2ff99f26b5e21665eb6cb3b686`。反例使用简化、格式完整且连续的commit JSON，仅隔离真实None-substage接口；没有声称该两步是合法门禁课程轨迹，也没有进行仿真。V/R真实可达性来自已读生产转移实现。全部producer夹具字节及五文件前后SHA保持不变；无Torch/model/env/PPO/pytest/GPU调用。

### 限定修复与复验要求

由原owner根按fix-loop保留before字节，先记录None-substage反例RED，再实现稳定、可区分V/R并可JSON往返的阶段键（具体命名归owner决定）；保留D/L substage及D22/D20跨度语义。测试须覆盖进入V、进入R、分批observe重读后继续、幂等、计数/累计成本/预算上界和producer完全只读。不改课程、奖励、预算或正式控制协议；reviewer不实现。

## 已核对的其他限定实现

### source-reload wrapper

- 真实入口委托A2 validator，执行前后核frozen observer_hashes且必须包含本wrapper；A2执行前后的产品freeze检查仍在。当前只完成静态接口审查，不能把此设计写成已执行GPU两load。
- 首次实际调用runtime.prepare_runner，来源唯一stop4817且使用生产迁移/裸model/load/save_verified loaded/ready；二次从首次ready原字节正式preload→新裸V07ActorCritic→PPO/manager→runner.load→save_verified loaded/ready，绝不第二次ETH transfer或学习。读取的model/完整Adam/normalizer/learning用原source tree_digest逐块对拍，保parameter-ID映射、groups和全部tensor；父链/source counters明确检查。
- 未加入seed/restore RNG/额外ppo.reset/env.reset/schedule调用。只包装原`_reset_ids`，原调用完成后存plain ids/context，并deepcopy快照；同源runner.load的四RNG时点审计被带出。`_observations(update=False)`仅读当时观测，不更新history/map或随机数。接受Qv07-001的schedule变化槽额外初始化，未称位精确或Q008完整隔离。
- 检查512真实role比例102/164/123/123、D25物理17、near15/16和easy1..14、generation73/gu0/phase_iteration2117、无tile_override、三维command范围及最终episode各缓存/history/map/零计数；pending reward reset队列仅记录原延迟reset状态，没有为“清干净”额外改奖励状态。
- first_environment/events先快照并写盘，关闭第一env、移除runner、gc/empty_cache后用弱引用断言env/model已释放才建第二512。两次调用原resource guard、另写测后resource证据；wrapper的第二runner没有学习/case_runner用途。实际释放、资源、RNG/cache值及runtime后续改动兼容仍须最终同freeze GPU验证。

### 显式正式shell

run_v07.sh默认不带execute，不因生成shell创建生产目录。num_envs512/eval8/rollout48/epochs5/minibatches4、命令.3–.6、初始std.30与.05–1边界、gamma.99/entropy.003/global48000和分阶段1000/12000/21000/6000/4000/4000、save100/diagnostic50/gate100、固定seed42/source路径SHA和manifest/acceptance/唯一output均显式。学习率.0003是原recipe断言，实际恢复必须保4817 Adam lr=7.593750000000002e-05；shell没有以该recipe断言重置Adam。真实runtime.resolve_request会把指定train请求解析为唯一源的resume合同；未添加fresh入口。未执行正式shell、未尝试绕过尚不存在的最终acceptance/manifest。

### 独立实际recipe parity tests

测试直接导入冻结v06/C1与native v07真实rewards/PPO/model，没有用同一v07函数同时作两端。18×4 reward frames覆盖全部非零terms、weighted totals、RewardState、reset-forward与success/failure冲突优先；GAE除两实现对拍还独立锚定timeout/terminal数值；实际完整actor同权重、同RNG排列、lr7.593750000000002e-05下执行20 Adam steps，比较全部model state、完整optimizer参数ID/groups/45状态、输出metrics与learning计数。此为有意义的数值保持证据；未冒充物理登顶或GPU表现，也不推出所有可能输入的数学形式证明。

### observer中已成立的限制和统计语义

只接受指定source/recipe/phase budgets及实际receipt4817/gen73gu0，拒未来native恢复段而不猜保持streak。输出必须在producer外；导入不Torch/仿真/控制/重启。commit从4818开始连续、聚合spent非负整数且sum=update，已有hash保留；先检查新批次缺口再输出。活动学习累计从118382592控制/473530368物理/96340Adam按每更新24576/98304/20推算；formal加原lost100，active48000/formal48100；阶段上界按当前宏阶段与之后剩余预算计算，不借已跳过F0/H余额。该结构在已测D25/D22片段正确，但F-ROOT-01使全阶段能力尚不成立。

保持判读明确summary_only，仅独立复算每速5×三速的failure、>=12告警、两轮streak停止，table/raw/retention一致，保留raw passed=false以及原required文案；“未告警”不改写成15/15成功，也不是case级独立gate。gate与retention文件各SHA双绑定，不只绑定其一。未建立Q008完整evaluation RNG隔离，observer不修改训练数据/门禁/终态。

## 验证证据及未重跑说明

已完整读root局部原文：recipe parity_03显示3passed/9.01s（上游jit warning），observer_green显示3passed/.02s，engineering_help显示真实CPU--help输出和同一warning。这些raw没有统一command/EXIT尾注，本报告据其可见stdout计数和root回执归属，不伪造独立shell exit字段。没有重跑同SHA的这两个套件、shell syntax或help。三原文SHA：

```json
{
  "root_recipe_parity_03.log": "4a4c9f1a40a9e3c2e2ec96fa8d3925992546324b00337f720bc0bb4114190e3b",
  "root_observer_green.log": "bf22e88f0e3a0e89eddb09d252b1661a48c9dfc4bd5aa177b5efca41f2700a18",
  "root_engineering_help.log": "b4bef2069cd12ac9a8c4d5b055f91136fc7888ade191dbfec9d41a28246eaa96"
}
```

本次只新增前述两分支JSON counterexample（工具exit0）和本独立报告，不新增功能tests、不改五个owner文件、不消耗工程学习或正式48000。

## Declined-to-judge／仍待验范围

1. 尚未稳定的B Q002控制桥接、manager/schema/恢复事务、window64/EMA/FIFO交叉绑定、晋退generation与重复应用边界；本包静态消费者追踪不等于B核心通过。
2. 两load四时点实际RNG摘要、reset槽次/全部缓存、非空Adam、512+8实际显存/host/disk、资源峰值、串行释放；wrapper尚未执行GPU。
3. 同最终source/test/docs/observer freeze完整CPU、五实际GPU阶段、probe50真实15首records和55→56恢复、45baseline/30保持500窗口及成本；不得用局部CPU替代。
4. 训练效果、最终停止状态、未来native续段observer协议、奖励或配方变化建议；均无本轮授权/实际数据支持判定。未来续段该observer会明确拒绝，不新增隐含支持。
5. F-ROOT-01修复命名与实现归owner；reviewer只报告要求，不自行修改。

## 文件绑定与本批修改

ROOT_OWNER_SHA256.json实际SHA `1072ed09b4bbcb233b8362f4ec39fc8d97635aafbb74dd79817cb6ea52b8c5ab`；ROOT_REPORT.md实际SHA `cfdaadabc9b99601f45a6f7b78c5748d532d0acbe60ebd715511825bd5a6d370`。

|文件|本次审查SHA256|
|---|---|
|logs/a2_ideal_stairs_v07/implementation/run_engineering.py|`9088ee0f3f47d3c6d95178dee1dea16556ff67ba86ef42a55830296f7545edf0`|
|logs/a2_ideal_stairs_v07/implementation/run_v07.sh|`47b8f7c3f9b91a1c2fb169f53f0744c18edf03fad332af8a9182fc57c8e104fe`|
|logs/a2_ideal_stairs_v07/implementation/coordination/v07_observer.py|`255ad13baf7219d4ad29383df6c38ffe1c0d02f3551460f652a3b59bf628ea7d`|
|tests/test_v07_recipe_parity.py|`dd582b7dd77d02f28d05c4a758d8c800fb1986453a11b7585449ce332ba337d6`|
|tests/test_v07_observer.py|`92cae5dc6e2967ab5a1faeb036fa1dbace7c212db299256edb1e1136a9ab2134`|

本批只新增TASK_ROOT_REVIEW.md；独立反例位于/tmp。没有修改ROOT产品/执行者tests、A/B owner文件或既有审查结论，也没有共享日志/Git写。root统一追加[logs/codex_changes.md](../../../codex_changes.md)，记录本包REQUEST_CHANGES、实际P2和轻量CPU反例。修后只限定复审finding及新diff，不重复已通过同SHA局部套件。


## R1限定复审：F-ROOT-01关闭

日期：2026-11-17T14:06:33.169383+00:00。本次只读ROOT_R1_FIX.diff全部新增差异、ROOT_REPORT R1章节、ROOT_R1_OWNER_SHA256及真实RED/GREEN原文，复验原两个纯JSON分支，不重复5项或旧局部套件。没有游走到未冻结B范围或修改任何产品/tests。

### 双verdict

- **Spec compliance：PASS_WITHIN_ROOT_R1。** F-ROOT-01/P2关闭，ROOT本包确认blocking由1降至0。
- **Code quality：PASS_WITHIN_ROOT_R1。** 本轮diff没有新增确认P0/P1/P2；原三个未改ROOT文件保持原SHA，observer只调整None-substage到phase的阶段键映射，新增测试只验证本finding。

当前逻辑在substage非None时保留原D/L阶段键，在None时使用phase，因此V和R得到独立字符串键，可JSON排序、写盘和重读。D22/D20跨度仍按原stage/generation边界关闭；source计数、预算上界、保持summary解释、producer只读和原control/课程路径不变。没有仅压制JSON异常或把V/R合并为字符串null。

### 实际复核证据

1. 已读真实先RED原文：新增one_batch和separate_reads两个参数化均因原None/str TypeError失败，2failed/3deselected，.08s。RED日志SHA `b5a37af4bebb6c509d0f7969c1f236199689dbb28f2841566c2eebfc0908b34e`。
2. 最新原文显示完整observer文件5passed/.03s，没有failed/skip/error/deselected条目；新增两参数覆盖D22→D20→V→R整批或分批重读、distinct V/R键、JSON往返、幂等、跨度关闭、源累计control/Adam和producer字节不变。GREEN日志SHA `0e96d7ca7406320840ee2e704cbada03e2fe4b6d6aea1d667b4f685b07e7e24d`。两raw没有统一EXIT尾注，沿初审同样如实归属可见stdout/root执行回执，不伪造reviewer重跑结果。
3. reviewer独立实际复验原V/None与R/None两个JSON分支，原先崩溃点均现成功：D25+V或D25+R独立计数，state JSON往返完全相等，重读幂等且audit进度无重复，producer字节不变。工具最终**exit0**。未运行pytest、Torch/model/env/PPO或GPU；简化JSON依然只证明字段/序列化接口，不称合法两步门禁训练轨迹。

独立回执：`/tmp/a2_v07_root_r1_independent_recheck.json`，SHA `e4d93e795405e0698bee71f8650fc53074fdaf8d719451172d55b127ad7e6ff1`。五owner文件在轻量复验前后均匹配原manifest加本次两文件overlay，未改三文件保持原SHA。

### 当前绑定与继续待验范围

- ROOT_R1_FIX.diff SHA `0f3b6a184854fe971b3c163b8bbd7d4f053aa22b4aba61e1a0d0840d232fe19c`。
- ROOT_R1_OWNER_SHA256.json SHA `f5c57e0f48685f7b109db6ef012a2e046a0b0adc8f5b807bea4a1d43579ac42f`。
- 当前observer SHA `225132a4681123fef9d3a337502d420e5012844dd4014f57c0bc746e7c246e5f`；当前observer tests SHA `b85fe3ed4b2f07242483924e729f55b0cbb066a8a83736ee60f32f9c49d2bd66`。
- 当前ROOT_REPORT.md SHA `79a15533ba4e34c3efea68b679d272dfbacbf7b2aaa32777dcd66fa74233aad1`。

原declined-to-judge清单继续有效：B核心/控制/事务/window64、同freeze完整CPU、真实五stage GPU/512双load/实际RNG缓存资源/计费及行为效果均未由本轮验证。未来native段仍明确不受本observer支持。ROOT模块blocking0不等于全局放行或允许reviewer启动生产。

本批仅保留历史并追加本审查文档和/tmp回执；未改五owner文件、共享日志、Git或任何训练状态。已交root统一追加[logs/codex_changes.md](../../../codex_changes.md)。
