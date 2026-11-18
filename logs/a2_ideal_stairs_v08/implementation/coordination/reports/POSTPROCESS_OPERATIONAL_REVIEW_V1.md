# v08 终态后处理队列操作审查 V1

日期：2026-11-18。独立 reviewer `/root/v08_final_reviewer`。

**结论：暂不可排队。Critical=0、Important=3、Minor=0；仅针对这份操作helper的blocking=3。** 不改变已通过的W5工程准入，不重开冻结训练源码，不需要中断当前formal。

只读完整审查353行 `implementation/coordination/post_formal_pipeline_20261118.py`，实际SHA `a85df561e89941aeb71410d7f5de872ffd23c2456c3cd7fac234e02b3d0d9166`。只针对进程组/单GPU锁读取既有job_controller；runtime/native验证接口沿用此前已审语义，未展开模型/训练代码。没有GPU、物理、学习、正式控制、源码/日志/Git改动，没有委派；仅临时目录/内存CPU mock与本报告写入。

## 已正确实现的边界

- `terminal_plan` 明确允许真实finished/exit0、formal、账本一致且无pending的两类SDS终态；operator/error/running/未知stop拒绝，cap验证对应预算用尽与边界已结算。cap失败不跑final2100，但仍保留8预设失败录像。
- `native_inputs` 核对唯一formal目录、最新durable checkpoint、实际SHA、v2冻结/病例、既有独立验收及baseline、native模型/Adam/费用/sidecar/journal，未改正式权重或来源。
- 子stage使用原生final/replay入口、updates0/seed42、8评估或1回放，沿用单GPU锁。输出或attempt已存在即拒绝，不自动重评。final2100真实有效分母完成但策略门槛未过时，仍记录policy_passed=false并继续交付固定录像，没有挑成功替换。
- 8预设case顺序与冻结final:E选择一致，包含0.5平地/顶停/E1/E2/E3 up_reverse/E4和E4的0.3/0.6。失败保留，最终root/独立结果复核仍被明确要求。
- 显式取消路径向它自己 `start_new_session=True` 创建的controller进程组发SIGTERM；已定向确认job_controller的worker未另建session，因此继承该组。代码没有向正式worker PID或正式control文件发信号。此为源码/依赖核验，不声称执行过真实取消信号测试。

## Important O1 — 重复启动可覆写已有队列状态

位置：helper `:240`、`:241`、`:349`。

execute先mkdir(exist_ok=False)，后取driver锁；已有PIPELINE会直接FileExistsError，但main catch仅凭PIPELINE.exists就调用event，向属于前一实例的events追加并将status覆盖为postprocess_blocked。重复启动或恢复命令可污染已完成结果的当前状态，也可与仍在工作的实例写同一status。没有重新执行GPU不等于没有修改历史证据。

独立CPU临时目录复现：先创建status为 `{'kind':'postprocess_complete','original':true}`，原样调用main；mkdir正确拒绝后，status实际变成 `{'kind':'postprocess_blocked','error_type':'FileExistsError',...,'no_automatic_retry':true}`。该实验不使用真实PIPELINE。

最小修复：先获取driver锁，再创建唯一新目录并登记本实例ownership；只有已成功拥有本次目录的实例才可写event/status。未取得锁或mkdir失败直接拒绝，不能向已有队列写任何内容。补重复启动面对complete/running目录及锁竞争的只读不变断言；保留“不自动重评”的原行为。

## Important O2 — E通过标志没有与已验证的原始门禁summary绑定

位置：helper `:112–119`；必要接口是 `native_inputs` 已验证的 `state['evaluation']`。

当前只看curriculum自报gate_history末两行passed与streak，就决定额外跑final2100。native validate_checkpoint确实重算每份引用sidecar/summary并核对账本，但它没有把cur.gate_history里的passed/streak与那些实际summary关联；所以“native完整验证”不能替代helper的这个最终准入条件。

独立纯内存反例：构造预算/学习计数及E历史末两次连续passed均满足的终态，但对应 `state.evaluation['gate:<当前update>'].summary.gate_pass=false`（420完整valid且无工程错误）；terminal_plan仍返回 `run_final2100=true`。此为最小决策层反例，不声称真实formal已发生不一致，也未篡改真实checkpoint。

最小修复：至少将最后两次E历史更新，逐一绑定到native_inputs已经原始重算过的实际gate reference：identity的kind/phase/update、冻结gate:E case SHA及完整420分母/0invalid/error/engineering_error=false、实际gate_pass=true，并核对与history一致。缺件、false或phase/update错配全部拒绝。若选择按原CurriculumManager规则只读重放全部阶段gate历史，可更完整证明A→E与预算账本；不要改冻结课程或重新跑门禁。补上述false/missing/foreign ref及合法完整E正例。

## Important O3 — 解码returncode0不足以证明无解码错误或完整录制

位置：helper `:225–235`。

目前仅查非空mp4和ffmpeg returncode0；`-v error`只控制输出级别，不设置遇解码错误即失败。recoverable错误写入log时仍可能返回0，而helper不检查错误log；即使一个视频干净解码但只有很少帧，也没有与该例control_steps核对长度。结果仍标full_decode_exit_code=0和最终媒体complete，可能把部分录制当成完整连续回放。

独立纯mock反例：8个临时文件、各例声称control_steps=1000；mock解码器每次写入非空“mocked recoverable decode error”并返回0。decode_videos实际接受8条，8份非空error log被忽略，结果中没有帧数校验。**这是helper契约mock，不是真实损坏视频/真实ffmpeg测试。**

最小修复：ffmpeg启用 `-xerror`，并拒绝错误日志或明确解析错误状态；提取实际完整解码帧数、fps/尺寸/时长并与原native replay录制协议核对。当前原生协议是1280×720、25fps、每2控制步一帧、初始帧以及必要的终态强制帧，因此无丢帧时应为 `ceil(control_steps/2)+1` 帧；时长按此帧数/25核对，可有格式时间基容差但不能掩盖缺帧。每个video路径应绑定对应case_id文件名，防止同目录另一条/重复视频充数。补错误日志、非零退出、短但可解码视频、fps/尺寸错配、完整偶/奇终态帧计数的CPU mock；无须运行额外GPU。

## 实际CPU输出

工具环境：`CUDA_VISIBLE_DEVICES=`、`A2_V08_WANDB=0`，A2 Python `-B`，先import helper（不调用execute真实路径）。临时目录均在/tmp，退出自动清理。正式目录及进程未触及。

原helper独立self_test实际输出：

```text
Ran 5 tests in 0.000s
OK
SELF_TEST_EXIT 0
```

5项覆盖cap单次通过、完整E两连gate、operator/error/running/engineering拒绝、partial/budget拒绝、未结算cap/非E拒绝。它们不覆盖本次发现的ownership、summary绑定或媒体完整性。

新增纯mock实际输出：

```text
EXISTING_STATUS_AFTER_DUPLICATE {'kind': 'postprocess_blocked', 'error_type': 'FileExistsError', ... 'no_automatic_retry': True}
CONTRADICTORY_E_SUMMARY_PLAN {'kind': 'curriculum_gates_passed', 'run_final2100': True, 'phase': 'E'}
MOCK_MEDIA_ACCEPTED 8 nonempty_error_logs 8 frame_count_checked False
```

## 后续边界

建议仅在这份冻结闭包外helper统一修以上三点并定向复审；不要中途修改src/scripts、训练数值或升级freeze，不改ENGINEERING_ACCEPTANCE。当前队列尚未运行，问题不要求停止formal。正式最终门槛/策略效果及未来真实视频仍由实际终态证据回答；本报告只决定操作队列能否排队。
