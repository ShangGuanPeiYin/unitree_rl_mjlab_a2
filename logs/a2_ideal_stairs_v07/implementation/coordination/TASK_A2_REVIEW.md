# A2限定独立审查：worker与五工程stage

> 最新限定结论（R1）：Spec compliance 与 Code quality 均 **PASS_WITHIN_A2_R1**；F-A2-01/P2 已关闭，A2 当前确认 blocking=0。下文初审 REQUEST_CHANGES 为保留的历史结论；本次不构成完整 v0.7 放行，真实 GPU 与 B 事务/恢复仍待最终同 freeze 验收。


日期：2026-11-17T13:17:58.689605+00:00；reviewer=/root/c1_reviewer，gpt-6.1-sol/high。完整读取TASK_A2_BRIEF、TASK_A2_REPORT及1515行/84541字节审查包；包SHA58d2a992f27e3ccc3195513b947c79a191f4a21e9c71b04d1b0791c70ffca6cb，审查与反例前后两脚本/两tests匹配A2_OWNER_SHA256。

## 双verdict及分级

- **Spec compliance：REQUEST_CHANGES。** 五stage主要接线符合SDS，但probe真实update50证明的完整性/来源绑定有P2缺口，尚不能据其返回值证明完整首回合原件属于本次probe。
- **Code quality：REQUEST_CHANGES。** 确认P2=1（F-A2-01两个表现）；P0/P1=0，A2 blocking=1。root已接受并交A2原owner按R1限域修复。
- 没有声称正常实际producer已经产出错误证据；没有执行GPU、正式学习或重跑owner16项。整个v07仍PENDING，A1原限定PASS不受本新finding改写。

## 已符合的范围与证据

1. worker使用native v07 bootstrap和模块，build_worker/main只委托B runtime，旧正式fresh/历史桥入口已移除。唯一4817迁移及完整resume守卫归B，不能由本审查推定端到端通过。
2. 当前case路由继承C1逐例kind/measurement/lesson/tile/vx检查，长期D25 closure能派生D22/22、D20/18、flat0/H001及final21；每批最多8、实际case数环境，没有以最大22作为最终目标。
3. D22 smoke实际构造8env并强制tile22、2次PPO；每更新计数差分和总768控制/3072物理/40Adam检查，逐更新资源记录。ETH初始化仅用于工程smoke/probe；正式worker不fresh，工程容器不作正式源。
4. probe计划R/L40、512+8、5warm+50measured+1恢复；首55设置evaluate=True，调用update50原件证明，55后裸模型load/save_verified loaded→单56更新→ready，计数按权威checkpoint55与累计56核对，无第二次ETH transfer。实际GPU未运行，改flag不等于已取得15物理首回合。
5. baseline从唯一stop4817原模型及normalizer恢复，D25/D22/D20各三速前5共45例，跨terrain的重复caseID用tile/vx/case组合区分；保持flat/H00共30，长期D25 closure测物理0/1；不计资格、streak或正式48000，不设行为成功率准入线或挑结果重跑。model前后digest不变检查覆盖全部state_dict。
6. 每真实评估批写ordered cases/first records/summary及各hash和actual/useful/extra控制物理成本；每批测原四资源门槛并先保留资源文件后拒绝异常。retention报告500步window_end实际覆盖数，不能把所有早失败当500窗成立。
7. 无execute只输出plan且不mkdir；execute拒source不完整，固定freeze读取source_hashes/closure/contract/SDS实际字节，运行后再核；目录存在和正式唯一目录/其子目录拒绝，leasefinally释放。最终tests/docs/observer额外map由根统一freeze/wrapper核验，不在A2脚本里假称已经完成。

## F-A2-01 — P2：诊断证明只自校验payload，未验证严格首终态和真实probe50来源

位置：train_a2_ideal_stairs_v07.py的probe_diagnostic_evidence。该函数读取runner.diagnostics唯一update50，实际验证sidecar文件与三个内容hash，这是必要的；但其restore_result(payload,payload['identity'])只证明payload对自身identity一致。额外检查仅update50、kind diagnostic、15例、target21及first_episode为非空dict。

没有校验首terminal所必需的env/case/tile/step/outcome/finite字段，也没有将source/model/optimizer/contract与本次probe真实50时点独立来源对齐。当前CPU正例夹具first_episode本就仅env_id/control_samples，并没有完整生产terminal证据；不能靠它证明严格消费者。

### 已实际复现的两个表现

- 将15条first_episode全部改为`{'arbitrary_non_terminal_evidence':1}`，重算记录hash并经真实evaluation_records写入合法sidecar。probe_diagnostic_evidence仍返回case_count15/费用证明，实际首终态关键字段全缺。
- 一个自洽sidecar的source_closure_sha256/model_sha256均为全0，甚至identity没有optimizer_sha256，函数仍返回成功；没有任何本次probe50的独立expected identity来拒绝它。

反例不是哈希未更新的损坏：文件绑定和三SHA全部自洽。缺陷正是“自洽数据”与“真实本次50完整数据”的证明层次没闭合。scope不外推正常runner已实际选择foreign sidecar或发生GPU错误。

### 最小修复验证要求

复用生产首episode验证口径，逐例核对完整terminal/env/case/tile/steps/outcome/有限值与成本平衡；缺字段、非终态、错槽、错tile和同帧不一致应拒绝。此为证据完整性，不改失败case判据或加成功率门槛。

identity必须来自真实probe update50现场的独立记录/冻结checkpoint与诊断主entry，检查本次source/contract/model/optimizer/task/gen/update/table等；不能拿payload自己的identity当expected，也不能把之后已训练到55的模型冒充50模型。原runner在诊断边界已有保存机制，可按owner设计绑定实际50字节，不需要新PPO或重评。

root已按fix-loop交原A2 owner先RED→GREEN。reviewer不写产品/tests。修后只限定复审此finding和新diff，保留本次原包/原source/反例；不为造日志重跑原16项。

## 反例与局部GREEN证据

- 反例：/tmp/a2_v07_a2_probe_independent_counterexamples.json，SHA2dd68a5fda436d3f72ced1b0f5c42b070d90c7da533cbc60a31eea14e9feaa12；CPU工具exit0。worker前后SHA41fc1556567332424eb5203088a639c4968af7758d4f72e0c48de5fa95e21d92；validator b03afac67df40b16f574b6c8576497cd7f403a66eaa5f9ccb0a4f47e4fc9574e。
- 方法仅用现有15条JSON夹具、真实sidecar pack/hash/写盘/读取；无model/env/PPO/pytest/GPU。它不是15个真实物理episode，不计工程学习。
- A2_GREEN_03原文已读：16passed、0failed/skipped/errors/deselected、20.50s、EXIT0，保留上游jit弃用warning。原文SHAe4043d803913a11d266e744e66c119dc0ed71c481341a5c231738b8a8630bf33。reviewer未重复该套件。
- 最后夹具每批4槽3步+1槽8步，首60/actual120/extra60；旧批次45/120/75夹具历史保留。两者均不代替真实GPU诊断。

## 尚不能判定项

- B runtime/continuation及runner/schema当前WIP，正式不fresh/唯一源、非空Adam/metadata、恢复保护期/队列/时钟和普通终态守卫必须由后续B包闭合。
- source-reload原stage只返回第一次prepare_runner；根observer按brief增强独立第二次512 load/reset/RNG/cache/release。没有该实际增强证据，不能把一次stage返回值称双load通过。
- probe首55与恢复56的实际资源/对象释放及监控仍待GPU。当前50样本明确是updates6..55窗口，不是连续峰值证明；根最终应按实际资源记录标注恢复阶段，不能在没有测量时声称峰值达标。本审查没有发现实际超资源证据，不擅自降env/改精度。
- 五stage真实GPU、23tile实际512+8、first15、55→56、45baseline、30保持500窗口、完整同freezeCPU/资源/成本均未运行；CPU16夹具和本反例不替代它们。

## 绑定与本批修改

A2_OWNER清单SHA78130003b95c2a69e076635011e8f10ce39edcf017e69ece48bc58a1013a4b54；实施报告bf93411390f787254cc040ae10ee1f6e681c50b679cfab7c41b26c7d09021ff2。原限定文件：

|文件|审查SHA256|
|---|---|
|scripts/train_a2_ideal_stairs_v07.py|41fc1556567332424eb5203088a639c4968af7758d4f72e0c48de5fa95e21d92|
|scripts/validate_a2_ideal_stairs_v07.py|b03afac67df40b16f574b6c8576497cd7f403a66eaa5f9ccb0a4f47e4fc9574e|
|tests/test_v07_worker.py|508fa5fa287d81faa89a84c1de27966dec9d273ad72890b791b134a18d1ec822|
|tests/test_v07_engineering_protocol.py|574f0f1e11f6097d29e5ed4ac154f8cc1068e3a7cb2b830df7b8ab3635615b09|

本批只新增本审计文件；未修改产品、执行者tests、A1审查或共享日志，未GPU/正式学习/Git写。root统一追加logs/codex_changes.md，记录P2、真实小CPU反例及修复待验边界。


## R1限定复审：F-A2-01与新增差异

日期：2026-11-17T13:34:34.345771+00:00。只审原finding、A2_R1_FIX.diff全部311行/20523字节及其中新增调用路径；未实现生产代码或执行者测试。新worker SHA `5d128dc1a039c1054f248d5ecbc3cdc19451f85ff9883f9e60948a32e5f48ff3`；validator与worker tests保持原SHA。diff SHA `804675dc8fbd60c9b87232b1236f8a4f9bcff15c31b08328e59eb3de8985022a`，五文件清单 SHA `d2d93306552391a83c0ae882ae9c780b903eb7918f8e623eaaab71c7d701ee6a`。实际五文件在本次轻量验证前后均逐字节匹配该清单。

### 双verdict与关闭理由

- **Spec compliance：PASS_WITHIN_A2_R1。** F-A2-01的严格首终态与本次真实P50身份绑定两个表现均已关闭。A2确认P2/blocking由1降为0；不据此宣告完整v07 ready。
- **Code quality：PASS_WITHIN_A2_R1。** 本轮diff没有另发现确认P0/P1/P2；现场捕获和消费者各自承担明确证据职责，没有加入成功率门槛、重评、PPO更新、初始化或随机数重置。

真实循环在`train_one_update()`已返回update50后、任何update51前调用capture。独立expected identity直接来自当时真实model.state_dict、完整Adam digest（保留parameter-id映射和param_groups）、learning累计计数、课程generation、当前source closure/contract及重新构造的冻结R/L40 diagnostic表。它不读取sidecar的identity作为expected。非空Adam与50×20=1000累计Adam计数都检查，捕获JSON及其文件SHA独立保存在runner上；主diagnostic条目深拷贝，不能因后续对象别名静默随改。55消费者使用该50捕获而非55当前模型，核文件SHA/source/contract、主条目全等、三证据SHA、表身份、records重算summary及sidecar文件绑定。

逐例消费者复用现行生产`_validate_first_case`，另将15条ordered records与冻结case逐项对拍：批slot/env_id、物理tile/lesson/frontier、phase/substage/seed/generation、首观测/terminal/control步数、原始与导出outcome/reason/诊断指标、实际pose/command、每批真实循环及费用平衡。pose/command按真实float32再沿既有`_same_case_scalar`核对，没有新epsilon。合法body失败照常被接受，不能将严格证据验证误解为要求成功。

### 实际证据与验证限制

已完整读owner原文：原两个反例的RED为2failed/exit1，SHA `5e8c1d72df8ed530161279b91e3a44e002cfa395390f4f7026c300aef45dbcd4`；最新完整两A2文件为**32passed、0failed/skipped/errors/deselected，24.07s、EXIT0**，原文SHA `7cba4814459ebaab19b2dcc54e9124f557f0e6ebd4af4af13b6a2102f62a625b`。两次名称含GREEN但实际17pass/15fail的夹具接口失败已在owner报告保留，未当作通过；最终仍保留上游jit warning。

reviewer不重复32项，仅实际运行5个轻量CPU消费者检查，工具最终exit0：

|检查|实际结果|
|---|---|
|真实tiny model/完整Adam在50捕获，随后改模型模拟55；15条均body失败|合法接受，仍绑定原50身份；首60/actual120/extra60控制步|
|捕获后修改主diagnostic条目|拒绝，主条目与现场深拷贝不符|
|三SHA自洽但first_episode任意dict|拒绝，首终态/slot不完整|
|三SHA自洽但foreign model identity|拒绝，独立expected identity不符|
|三SHA自洽但缺optimizer identity|拒绝，独立expected identity不符|

实际回执：`/tmp/a2_v07_a2_r1_independent_recheck.json`，SHA `631b98b3a6a1cf73f6158a7be4f5b90f4e914a894dccafad80ab93f87e134691`。此处使用受控terminal JSON与tiny torch模型/1000 Adam steps，**没有环境、PPO、GPU或pytest套件运行**；不冒充真实50更新或15物理episode。首轮原反例文件及初审包/报告内容均保留。

### A1单行必需依赖补充审查

按root批准，spec.PRODUCT_MODULES required tuple仅追加`window_evidence`。直接用原A1 spec快照对当前字节替换比较，确认除此一行外完全相同：before SHA `4a19735397b11bd3f9c25cceed6c47f3b7babcdc45a9cb64e7465ca991de116b`，after SHA `6480567fdd3c6fd9265d102e54f742f56276c6f2e3b32cf9a9b60e89fabf2253`；diff SHA `bc78f3c21299afdfebaee0999aece9ff9b5618284dad86739c4f3d161ec3abef`，before/after绑定文件SHA `33e47e2fbb3b8220d52301bdab132c60de24eac69ce52bafde991cf6972a2929`。

该行把SDS要求的64窗原行证据模块加入必需闭包，缺模块时应fail closed；没有改几何、预算、PPO、奖励、速度、出生或终止规则。**该单行spec compliance/quality均PASS_WITHIN_DEPENDENCY_DELTA**；不签window_evidence模块实现正确性，该模块归后续B包。原TASK_A_REVIEW.md及A1几何结论保持不变，原SHA仍只对应原交付。

### 继续保留的验收边界与本批修改

B manager/schema9/runner bounded事务、window64持久化、真实首次/二次512load及保护进度/队列/时钟恢复，仍待后续稳定包。五stage真实GPU、512+8、P50真实15首records、55→56与恢复资源、45baseline、30保持500窗、同freeze完整CPU和成本均未由本轮提供。禁止由本轮CPU通过推断行为效果或正式学习准入。

本批只修改本review文档（保留初审追加R1）和写/tmp独立回执；生产/执行者tests/A1审查/共享日志未改，未GPU/正式学习/Git写。root按本批回执统一追加[logs/codex_changes.md](../../../codex_changes.md)。
