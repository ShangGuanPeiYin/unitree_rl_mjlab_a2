# B03独立限定审查：纯policy及完整60例输入

> 最新限定结论（R1）：纯policy的spec compliance与quality均PASS_WITHIN_B03，F-B03-01已关闭、B03 blocking=0。下方初审REQUEST_CHANGES原文保留；manager/runner事务与整个v07仍PENDING，详见末节。

日期：2026-11-17T13:09:03.152306+00:00。Reviewer=/root/c1_reviewer，gpt-6.1-sol/high。SDS SHA303ae40a5c5e3fe82f9f0df3aae8d153136e016439087c8db9701aeb17197aca。范围仅regression_policy及完整payload夹具/单测，不验manager/schema/runner集成；reviewer未实现该模块或修改执行者测试。

## 双verdict

- **Spec compliance：REQUEST_CHANGES。** 正常完整输入下的谨慎退级规则符合SDS，但持久化last receipt/decision一致性不足，未满足恢复后不能错误消费/错误退级的既有要求。
- **Code quality：REQUEST_CHANGES。** 发现1个P2恢复state一致性缺陷（两个具体表现），B03 blocking=1；P0/P1=0。本报告不是整体v07最终放行。
- root已接受该具体P2，并交原owner B按fix-loop修复。正常producer是否会自行生成此错误未证实，不能冒称已发生训练错误。

## 来源与版本漂移

完整读取TASK_B_BRIEF、TASK_B_REPORT的B03章节，以及281行/19627字节review package。包SHA30f2d86aec7496a17c33886aad359818f88cbac440396bca25b5ed40c246c37f；包内policy SHAe0e2848797bb35167d8728c2edc70dc30fc5663fe34366fcffd9b5af82577c82。

审查时实际policy已更新为a38362d5e28712bf439dfe5f936ffc7c88326b70b05d568021a67d7b54fafd20，增加formal global%100及last_decision.identity检查。已告知root/B，不能将旧包当当前完整冻结。下面反例严格绑定实际a38362…源，运行前后SHA相同；不把新增检查误报为仍缺失。

## 已符合的接口与边界

- new_state/reset_generation/validate_state/apply_validation纯数据变换；不运行模型、读磁盘选历史或消耗RNG。阈值集中读取native spec。
- 完整评估首先经evaluation_records.restore_result复算三SHA、当前task与caller expected的model/optimizer/source/contract身份、训练statistics_unchanged及summary重算；再检查三速各20=60例、首terminal/env/tile/step/outcome/成本平衡。diagnostic/retention/旧generation/缺证据不按0成功入队。
- 当前新增global100整边界guard与相邻100约束相容。保护看generation_updates<1000，12/13边界、rolling最近三、Δ>=6继续且保留历史、Δ<6按正确前一级或最低停、gap打断、正常重复fingerprint和同key冲突的主路径均符合SDS。
- 原24项GREEN是owner对这些正常及坏payload路径的实际局部证据，未涵盖本次持久receipt/decision内部不一致。reviewer未重复24项套件。

## F-B03-01 — P2：恢复缓存decision和last receipt未与当前队列语义绑定

位置：validate_state的last_applied/last_decision分支，及apply_validation的duplicate返回分支。当前只检查缓存outcome在枚举、decision.identity==last transaction key；有队列时只比queue末update相等。没有验证decision outcome/target/success_counts/improvement/protection资格与实际队列及完整任务身份一致。

### 反例A：一次坏评估被缓存重放成退级

1. 使用现有完整60例JSON夹具、update5900/gen73/D20、generation_updates1083、成功0，合法apply返回continue，队列只有1条。
2. 保留完整payload与last_fingerprint不变，仅将持久last_decision改为outcome=regress、target_substage=D25、success_counts=[0,0,0]、improvement=0。
3. 实际validate_state返回True；重放同payload/同expected identity返回duplicate=True、regress/D25。此时既没有三个坏评估，还跳过应退的D22。

根因不是外部payload三SHA失败，而是内部缓存decision被作为已可信决策直接返回。修复验证应同时覆盖合法目标D22但不足三条的伪regress，不能只加目标名称检查而遗漏触发条件。

### 反例B：同update的异代/异任务receipt冒充当前队列末项

合法队列仍为gen73/D20/update5900；将last_applied改为gen72/L/L40、table SHA全0，并同步last_decision.identity，update仍5900。实际validate_state返回True，因为只比较队列末update，而未核对整个last key与queue末任务/代/table身份。

reset_generation允许把旧代receipt作为历史证据保留是合理接口，但有当前非空队列时，旧代/异任务receipt不得冒充该队列最后已应用项。不能以“允许历史receipt”放宽当前队列末身份一致性。

### 影响与修复边界

这两例需要提供内部不一致的恢复state，不代表正常未损坏生产流程已出现该错误。风险是state validator自称可接受该恢复状态，随后duplicate路径可能返回不满足SDS的课程动作。应由B加强缓存decision结构/语义与receipt/queue的一致性，非法状态应拒绝；或让重复返回由已校验实际结果和可靠持久上下文确定，不能相信任意缓存动作。

这是SDS已要求的恢复一致性修复，不是改低成功阈值/保护期/课程配方；reviewer不指定实现布局，不自行改产品。跨真正generation清队列后如何保留旧receipt且不重复apply，应由B和后续manager/runner事务证明，不能单靠本纯模块claim。

## 具体反例证据

/tmp/a2_v07_b03_independent_counterexamples.json，SHA bbf659abfe8fd0d9a736eee09f2cc5b757b32805512ab654813a0e37e6c1b590。记录当前policy SHA、两个输入状态差异与实际结果，最终工具exit0。方法为CPU现有完整JSON夹具，不构造model/env/PPO，不pytest、不GPU；不是重跑owner24项。

首次探索命令exit1，是第二个旧形式反例撞到B新加的decision.identity guard；该guard确实有效。最终反例B明确同步该identity后仍被接受，未把首次失败当产品缺陷。第一例在两次输出均复现。没有预设原生stdout日志，工具原输出及上述事实JSON为证据，不伪造实时日志。

## 不能判定项与后续限定复审

manager对保护/晋级/旧EMA路径、顶层schema9 state、真正非空Adam迁移、window64原件、sidecar/晋退checkpoint异常、cross-generation幂等及同freezeCPU/GPU均未包含本包，不作结论。

B修复后需交R1差异、新policy与测试SHA、针对F01的真实RED→GREEN原输出/exit。reviewer只复核finding及新diff可能破坏的相邻分支，不重跑已过24项，不用本次纯JSON夹具冒称任何物理工程验证。当前包与a38362版本的REQUEST_CHANGES历史应保留。

本批只新增本独立报告；未改生产/tests/共享日志。root统一追加logs/codex_changes.md，记录P2、当前vs包SHA、CPU具体反例/结果与整体未验边界。


## R1限定复审 — finding关闭，集成待验

时间：2026-11-17T13:21:36.972028+00:00。实际policy SHA c0ae8ca976501d4050f9377222b47f743c4d88c9d0c923dddee6cdaef208c8e4；B03_R1_FIX.diff SHA df836fdbd4b0b5e455c28c7e4df5d5569b31915b103700cfbe865726b0f31def。修改前a38362字节快照保存在B03_R1_BEFORE，原反例与初审结论不删除。

实际修复删除持久last_decision冗余动作缓存；state严格字段集拒绝注入旧动作字段，duplicate只返回continue/targetNone且不改state。queue末项与last_applied按update/gen/phase/substage/kind/table六字段全等，旧代历史receipt仅能随空新队列保留，不能冒充当前队列末项。

已读新增3项测试及实际RED3failed/27deselected，RED日志SHA9ec6490a739c4409045a8d1bee70f38610aed2aed9e8ffe76966c1396e408557；随后policy/manager/migration相关52passed、0skip、13.57s，原日志SHA131140efffc36658b84570530a23370f5f8b02376103e9ef30fffdd068f88383。该52是owner局部证据，不代表manager故障事务已验。

reviewer只作finding相关CPU JSON复核：旧伪decision字段拒绝；同update异代/异任务receipt+当前queue拒绝；一次合法消费重放无动作且state不变；真正第三坏评估仍正常D20→D22，重复该第三评估不再输出第二次退级；合法newgen74清队列并保留旧receipt通过。实际工具exit0，未重跑24/52项或执行model/env/GPU。

独立限定复核事实文件：/tmp/a2_v07_b03_r1_independent_recheck.json，SHA7670135c49416c740c7033d4d59598b29892f2ffe32e2e24ab9262f4f847cfec。

**当前双verdict：spec compliance=PASS_WITHIN_B03，quality=PASS_WITHIN_B03；原F-B03-01/P2关闭，B03 blocking=0。** 本结论只对上述纯policy SHA及其数据状态成立，不表示整个项目ready。

重要集成边界仍未关闭：如果policy已经记录last key而真正regress/stop尚未durable，duplicate无动作不能丢掉应执行动作。B已承诺下一包bounded curriculum_transaction持久poststate/events并在缺应用证明时阻止下一PPO。要用真实异常checkpoint/load验证该切口，不能凭本policy通过推出它已完成。本reviewer后续仍审manager/schema/runner窗口与晋退事务。

本批仅更新本审计文件并保留初审；未改产品/tests、A1审查或共享日志。root按新SHA统一追加codex记录。
