# 执行A2：worker与工程stage

日期2026-11-17；SDS SHA303ae40a5c5e3fe82f9f0df3aae8d153136e016439087c8db9701aeb17197aca。依据主协调正式TASK_A2_BRIEF扩展的单owner：仅`scripts/train_a2_ideal_stairs_v07.py`、`scripts/validate_a2_ideal_stairs_v07.py`及`tests/test_v07_worker.py`、`tests/test_v07_engineering_protocol.py`。不改B runtime/迁移/runner/curriculum/recovery/run/evaluate/test/bootstrap；不改已交A1报告与SHA包。共享logs/codex_changes由根按每批消息统一追加。无GPU、无正式学习、无push、无子agent。

## 批次A2-01：先RED、native worker与五stage实现

- 目的：保留原C1真实环境、长期case closure逐例路由与55→56裸load计算路径，补v07所需D22工程与45例基线；唯一正式stop4817迁移留给B runtime。
- 首先新增两个真实行为测试文件；实际运行CUDA_VISIBLE_DEVICES=''下13 failed/0 errors，exit1，全部由于两入口尚不存在；完整命令/stdout/exit在`A2_RED_01.txt`。这里不是skip或依赖缺失。
- train入口提供build_environment、case_environment_kwargs、build_case_runner、build_telemetry、build_measured_resource_guard、probe_counts、probe_curriculum、build_probe_worker；这些从旧worker选定函数继承，改为原生v07导入。build_worker/main只lazy委托B runtime，未提供fresh正式路径。环境按当前case tile/lesson/protocol重建，旧D25启动closure不会覆盖D22、D20、保持0/1或最终21。
- validate入口提供smoke/probe/source-reload/baseline/retention-protocol。无execute只输出JSON plan、不创建输出。execute先拒绝不完整source closure并验证固定freeze/v1/V07_SOURCE_FREEZE.json的source_hashes/source_closure_sha256/contract_sha256/sds_sha256，随后原资源门禁与单workerlease，执行后再次核freeze。测试/doc/observer额外map由根统一freeze/wrapper核验。工程输出拒绝已存在目录及正式唯一目录，不覆盖旧证据。
- D22 smoke真实8环境、2 PPO更新、48 rollout、5epoch/4minibatch、4物理子步；该阶段独立ETH初始化只用于工程，不进入正式恢复，也不能产正式可resume容器。8/2/rollout等来自spec.SMOKE_*/PPO_*，每更新实际PPO计数增量与合计768控制/3072子步/40Adam均核验，实际tile必须全22。
- Probe真实512训练环境、评估上限8、5warm+50measured+1recovery=56更新；首55在R/L40，保留现行PPO/模型/出生/命令；首次工程初始化继承原ETH方式，55后新建裸模型并正式load，save_verified loaded/ready，禁止第二次ETH初始化，累计55→56费用与单步增量校验。正式48000账本不消费。
- 与旧worker必要差异：旧probe首55设置evaluate=False；v07依SDS改True，让update50真正跑完整15例诊断。55保存前必须从runner.diagnostics找到唯一update50，并读实际Q010 sidecar、校验文件SHA/三证据SHA/完整摘要，要求15条非空first_episode、diagnostic身份、最终tile21与物理费用平衡；缺失或篡改不能返回成功。不把改flag当实际GPU证明。
- Baseline载入唯一原stop4817模型及normalizers，不迁移课程、不学习：D25/D22/D20物理17/22/18×三速各5，共45例；flat/H00各三速5，共30保持协议。模型前后state_dict digest必须相等。沿同一生产case runner，不加成功率准入线、不重跑抽签，声明Q008完整评估RNG隔离尚未建立。
- 每真实评估批保存cases、原始first records、summary及三SHA/文件SHA；身份是terrain+vx+case_id，跨terrain相同caseID合法。首回合控制/实际全部控制/额外控制与对应4倍物理子步分账；summary从本批实际records计算。每真实批后测原4资源阈值，不降低env数/精度；失败保留资源原件。
- source-reload依根确认先调用B runtime.prepare_runner一次，核4817/gen73/0学习并保存source_reload.json；根observer wrapper增强实际独立二load/reset/RNG/cache proof。A2不另造迁移入口或seed回卷。

首轮实现局部验证`A2_GREEN_01.txt`：11 passed、2 failed、0 errors，exit1；两失败均为B尚未创建continuation_v07模块，影响baseline plan，不添加fallback或skip。

## 批次A2-02：诊断每批资源与实际sidecar消费

- 新增probe_case_runner包装实际case runner，仅工程probe使用，保留原returns/supports_batch；每真实diagnostic batch后保存资源原件与首/额外物理计算费用。该helper先真实RED1 failed（缺接口，`A2_RED_02.txt`），随后实现。
- 追加CPU测试使用生产evaluation/evaluation_records构造并实际写入15例Q010不可变sidecar，再由probe_diagnostic_evidence读盘校验：可读回15例与45首回合/120实际/75额外控制样本；改raw、删raw、缺update50全部拒绝。这里使用受控CPU记录夹具验证消费者，不冒充15个真实物理回合。
- 资源测试用受控hardware snapshot替身验证真实写盘与门禁：host4095MiB失败并保留原资源JSON；阈值为原4096MiB，没有改生产资源限制。
- `A2_GREEN_02.txt`：CUDA_VISIBLE_DEVICES=''、OMP_NUM_THREADS=2、MKL_NUM_THREADS=2，pytest两A2文件但明确`-k 'not baseline_cli and not baseline_identity'`；14 passed、2 deselected、0 skipped、0 errors，exit0，9.99s。两baseline项等待B唯一源模块，未用skip掩盖。

## 当前未验证

B完整迁移/runtime模块与全部scripts尚未齐，尚未最终完整A2 GREEN。未执行真实GPU五stage、512+8训练、实际update50物理15例、55→56 GPU保存恢复、真实stop4817的两个512load、45baseline物理评估、30保持物理协议、正式首4818、资源峰值或爬楼效果比较。上述只在根统一freeze/完整CPU后由根执行，不以CPU夹具替代GPU工程证据。

## 批次A2-03：最终限定CPU交付

B continuation_v07模块落盘后，无fallback/skip地解锁baseline两项。最后一次A2完整命令为：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/xxs/anaconda3/envs/a2/bin/python -m pytest -q \
  tests/test_v07_worker.py tests/test_v07_engineering_protocol.py
```

实际结果：**16 passed、0 failed、0 skipped、0 errors、0 deselected，20.50s，exit0**。完整原文为`A2_GREEN_03.txt`；仍有同一个上游torch.jit.script弃用FutureWarning，没有压制。测试实际验证了CLI baseline无execute输出45例plan且不mkdir、三terrain composite身份/原表和选择SHA、D22 smoke真实环境参数计划、当前case路由和第二例损坏拒绝、probe55+1累计计费/最终21课程、完整15条Q010 raw sidecar读取与破坏拒绝、每batch资源证据写盘与4095MiB失败、实际/首回合/额外费用分账及不完整source freeze拒绝。

本批将工程8/2/rollout/PPO/三档速度等从已有spec派生，实际数值不变，避免第二份配方定义；修正文档SDS章节引用，不改变计算路径。另将CPU Q010夹具改为每个5槽批次4槽在3步终止、最后1槽在8步终止，模拟其余槽继续等待：合计首回合60、全部物理控制120、额外60。此前A2_GREEN_02夹具为45/120/75，仅验证代数分账；旧原文保留。新夹具不冒充真实物理评估。

当前A2两生产脚本与两个测试的SHA见`A2_OWNER_SHA256.json`。A2限定实现/CPU交付完成，供根生成差异包和独立审阅；完整项目CPU、同一冻结GPU五stage、root增强的独立二load、实际模型/课程/资源验收及正式学习仍未验证，不能由这里16项替代。B迁移正例/全恢复仍由B继续，不因模块已落盘就推定端到端通过。

共享修改记录：[logs/codex_changes.md](../../../codex_changes.md)由根按本报告各批消息追加。

## A2 R1：独立P2证据守卫修复（2026-11-17）

独立review确认先前proof守卫缺口，而非确认真实GPU producer已经造错记录。独立原件`/tmp/a2_v07_a2_probe_independent_counterexamples.json` SHA`2dd68a5fda436d3f72ced1b0f5c42b070d90c7da533cbc60a31eea14e9feaa12`证明：(1)合法重算三SHA后，15条first_episode只是任意非空dict仍通过；(2)自洽payload的model/source全0且无optimizer身份仍通过。原`restore_result(payload,payload.identity)`只做自校验，无法证明当前实际probe50。根指派原owner限定修复，未扩大到配方或行为门槛。

先保存原两脚本/两tests到`A2_R1_before/`，原SHA与A2交付一致（worker41fc1556…1d92、validatorb03afac6…574e）；原A1/A2审查包、报告及SHA文件均保留。实际先写上述两个反例并运行：`A2_R1_RED_01.txt`=2 failed、0 errors、exit1，两项均因原守卫DID NOT RAISE。

修改前：55读取诊断sidecar，使用payload自己的identity恢复，检查15条first_episode仅非空和计费平衡。

修改后：

- 真实`build_probe_worker`循环在`train_one_update()`返回update50后立即执行独立现场capture，然后才允许update51。capture直接读取当时真实model.state_dict、非空完整Adam state_dict、学习计数、当前source/contract、冻结diagnostic案例表、当前generation及主diagnostic条目，计算model raw tensor SHA/完整Adam digest，写入`probe_boundary50.json`并保存文件SHA。它不从sidecar读取“expected identity”，也不使用55时模型。没有增加学习、重初始化或RNG操作；这属于根允许的现场独立observer捕获方式。
- 55消费者必须持有该独立P50文件绑定，检查实际文件SHA/source/contract和主diagnostic条目与capture一致，随后用独立P50 identity恢复sidecar，主entry的三证据SHA必须与实际result完全相等。缺P50捕获、主条目替换或foreign/missing model/optimizer/source/contract/task均不能靠自洽payload获得通过。
- 每条原始记录复用生产`runner._validate_first_case`，要求真实terminal的计数、布尔终态、有限指标、合法活动/停滞计数与终态reason；额外按冻结case对拍顺序、当前批slot/env_id、tile/lesson/frontier、phase/generation/seed、首观察步/terminal_step/control_samples、case初始pose与执行命令、顶层outcome/reason/诊断指标。命令/pose先按环境本身float32表示，再沿既有`_same_case_scalar`口径交叉检查，没有新增epsilon。相同批的真实物理循环数必须一致，首回合与额外物理成本继续平衡。
- 合法失败记录仍通过：正例全部为body失败，成功数没有准入要求。未新增奖励、PPO、速度、出生、课程或资源设置。

新CPU正例使用受控terminal exports，以及真实tiny torch模型和1000次Adam steps（计数表示50×20）；在50时独立现场capture后直接修改当前模型模拟55，P50 proof仍能通过并保留捕获的50 model/Adam身份。这里不是实际512环境/PPO物理50更新，也不冒充GPU15例。13项自洽重算hash后的破坏覆盖任意raw、foreign model/source、缺/错optimizer、错env/tile/step/outcome/command/pose/generation及主entry evidence冲突；旧缺独立capture反例也保留。

验证原文与失败不隐藏：

| 记录 | 实际结果及原因 |
| --- | --- |
| A2_R1_RED_01.txt | 2 failed、exit1：原守卫实际接受两反例 |
| A2_R1_GREEN_01.txt | 17 passed、15 failed、exit1：新CPU fixture的source_hashes替身漏root关键字签名；未修改生产守卫放宽 |
| A2_R1_GREEN_02.txt | 17 passed、15 failed、exit1：tiny nn.Linear不具备完整策略training_state_snapshot所需std接口；改夹具直接用真实tiny model/Adam hash与生产compare_snapshots，未伪造策略接口 |
| A2_R1_GREEN_03.txt | **32 passed、0 failed、0 skipped、0 errors、0 deselected，24.07s，exit0**；完整两A2文件，CUDA_VISIBLE_DEVICES=''、OMP_NUM_THREADS=2、MKL_NUM_THREADS=2；仍仅上游torch.jit.script FutureWarning |

原16项A2套件已在本次完整32项中覆盖。GPU真实50诊断15例、真实55→56恢复、资源峰值、正式学习仍未执行；R1 CPU无法替代这些证据。限本轮fix差异见`A2_R1_FIX.diff`，当前五文件绑定见`A2_R1_OWNER_SHA256.json`，供独立复审。

另按B明确实际依赖和根批准，A1 spec只在PRODUCT_MODULES required tuple追加`window_evidence`，确保SDS5.4真实64原行证据模块成为必需闭包；无任何数值或逻辑改动。原spec SHA`4a19735397b11bd3f9c25cceed6c47f3b7babcdc45a9cb64e7465ca991de116b`保存在`A2_R1_before/A1_spec.py`；新SHA`6480567fdd3c6fd9265d102e54f742f56276c6f2e3b32cf9a9b60e89fabf2253`，仅一行diff在`A1_SPEC_REQUIRED_WINDOW_R1.diff`，独立before/after绑定在`A1_SPEC_REQUIRED_WINDOW_R1_SHA256.json`。原A1审查包/报告不改，未重复运行未变的geometry套件，根安排该行限定复审。

## Qv07-002：仅追加control裁决冻结文档（2026-11-17）

主协调指派spec owner将新主决策补充加入FROZEN_DOCUMENTS。实际读取`V07_CONTROL_PROTOCOL_RULING.md`并核SHA`dad4580188773b0345f3f741b5f8a3d9f988652d7a257c8e2205dcd1d23adbd1`，仅新增该文档三元组；未改原SDS、数值、课程、控制实现或B文件，也未猜测未来control模块名称。B明确实际helper名之后才追加required tuple。

spec修改前SHA`6480567fdd3c6fd9265d102e54f742f56276c6f2e3b32cf9a9b60e89fabf2253`，修改后`faa3d2b2d69c50c8dc4df1891b919176f3d81b820681c1f322ad113c8e1f15d0`。旧源保留为`A2_QV07_002_SPEC_BEFORE.py`，小diff `A2_QV07_002_SPEC.diff`，before/after绑定 `A2_QV07_002_SPEC_SHA256.json`；原A1/A2包与旧报告均保留。

验证使用显式`CUDA_VISIBLE_DEVICES=''`的a2解释器，native bootstrap后导入spec，确认新FROZEN_DOCUMENTS项唯一、实际文档SHA匹配、原SDS SHA仍303ae40…、code0.7.0/schema9/48000保持。结果exit0，原件`A2_QV07_002_SPEC_VERIFY.json`。这是限定文档绑定验证，没有重复geometry套件或执行GPU/control流程；control语义与真实runtime验证仍归B和最终独立验收。

## Qv07-002实际helper required闭包（2026-11-17）

B明确实际文件名`operator_control.py`并落盘后，主协调指派spec owner追加required。为避免运行期source漂移，先遵守B真实runtime CPU检查稳定窗口，等B报告session已exit后才执行本最小delta；没有中途改其测试闭包。

实际仅在PRODUCT_MODULES追加`operator_control`（SOURCE_ENTRIES从tuple自动派生），不修改任何B实现、原SDS或两个ruling。spec前SHA`faa3d2b2d69c50c8dc4df1891b919176f3d81b820681c1f322ad113c8e1f15d0`，后SHA`c79a0afec8daa1f8ab67e3e7d9991997c49d5f4c77a8ed83ef6891b6089e07c0`；旧源`A2_Q002_OPERATOR_REQUIRED_BEFORE.py`、最小diff `A2_Q002_OPERATOR_REQUIRED.diff`及`A2_Q002_OPERATOR_REQUIRED_SHA256.json`保留。此前所有A1/A2/R1包和报告保留。

验证只运行显式CUDA_VISIBLE_DEVICES=''的native import/闭包检查，没有重跑geometry或执行GPU：确认PRODUCT_MODULES与SOURCE_ENTRIES内helper各唯一，实际source_hashes覆盖helper且SHA为`9f752f6e5744a354c6fb01bbf3237ebc6e36192df2a2b65eda326e8d7169848b`，实际pending为空、源闭包94文件；在临时隔离root仅移除该helper时，pending精确返回其必需路径，从而缺失会fail closed。原SDS实际SHA303ae40…和Qv07-002冻结dad458…不变。结果exit0，原始结构化证据`A2_Q002_OPERATOR_REQUIRED_VERIFY.json`。完成后立即通知根/B新spec SHA，后续正式freeze使用该版；control行为的真实集成验收仍归B与独立review。

## 修复后freeze v2路径（2026-11-17）

根确认B CORE R1 affected57/57 GREEN且无在途CPU后，指派仅将validator的FREEZE_PATH从`freeze/v1/V07_SOURCE_FREEZE.json`改为`freeze/v2/V07_SOURCE_FREEZE.json`，使修复后的新闭包读取新freeze，保留v1原件。实际只改这一处路径；默认`gpu/v1/<stage>`输出逻辑仍原样，根后续显式传`--output-dir gpu/v2/<stage>`。未改任何配方、B核心、SDS或ruling，未建正式目录。

原validator SHA`b03afac67df40b16f574b6c8576497cd7f403a66eaa5f9ccb0a4f47e4fc9574e`的完整源码保存为`A2_FREEZE_V2_BEFORE.py`；新SHA`55158db282850b881fb46fc8890b58b8e41338510b074a0e8e616099eac61ad5`。限定小diff `A2_FREEZE_V2.diff`和前后绑定`A2_FREEZE_V2_SHA256.json`保留。

CPU验证显式CUDA_VISIBLE_DEVICES=''，native导入validator后核实际FREEZE_PATH为v2；AST赋值常量也为v2，并验证完整源码只发生该一次替换、source_hashes覆盖validator自身新SHA（94文件），CUDA未初始化。结构化证据`A2_FREEZE_V2_VERIFY.json`断言通过/exit0；仍仅一个上游torch.jit.script FutureWarning。按根要求没有重复已过32项、geometry或GPU；后续全native v2套件与GPU由根统一冻结后执行，不因这个路径检查推定整体已通过。完成后立即通知根/B新SHA并暂稳源码。
