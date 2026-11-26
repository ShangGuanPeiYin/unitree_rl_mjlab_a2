# A2 v1.0 runtime implementation — 2026-11-26

Owned additions: `runtime.py`, `checkpoint.py`, `resources.py`, `scripts/run_a2_v10.py`, `tests/test_v10_runtime_core.py`; follow-up corrections in owned model/replay/learning tests. Parent owns project change log and Git.

## Flow and implementation

`smoke` runs real nominal physics, all six roles, same-snapshot student/teacher inference, measured contact/rewards, deterministic student actions. `worker` executes explicit D1/D2/D3/P0/P1 phases without automatic gate transitions. Every full rollout is 48 control steps and survives episode boundaries; all completed physical samples are retained, terminal GAE masks reset and only the real final nonterminal observation bootstraps. A gate boundary exits only after the current complete update and state save. PPO keeps raw sampled actions/log-probabilities separately from executed actuator actions. Stair teacher construction always reloads the SHA-identified original S2 source, including on D2/D3 resume.

Atomic checkpoint schema is `a2-v10-state-v1`, binding SDS, recipe, both teacher hashes, physical contract, fixed N, all model/optimizer/std/normalization states, replay/count/RNG, physical episode restart metadata, controller state, external costs, and frozen keep model. Resume creates new physical episodes while preserving the learning state and role rotation. Checkpoints save on crossing each 2,457,600 confirmed learning samples after a full update, rather than relying on divisibility; boundary checkpoints are separately retained.

`BudgetLedger` implements planning ruling Q1: fsync append-only pre-reservation, unique idempotent step identity, executed marking, conservative uncertain reconciliation, one in-flight step, no refund/replay, and independent completed-update sample events. Charged cost is executed+uncertain; only confirmed executed samples drive gate/save thresholds. Incomplete rollouts have no update events and never enter GAE/gradients. Process-local validated offset/identity caches read only appended journal events; fresh processes replay and verify the entire journal. Reads and mutations lock the file. The report includes charged/executed/uncertain/updated, and checkpoint recovery cannot decrease external costs.

Physical identity excludes world/terrain body inferred inertias and includes only owned robot body/joint/dof/actuator/collision data, dynamics, gravity/solver/integrator settings, and fixed control/observation contract. Terrain identity is separate. Raw actions retain the existing no-pre-clip path; no ±10 clamp was introduced.

`probe` spawns a fresh subprocess for each N/stack. It tries N=1024, then 512 on failure; each candidate must pass full DAgger and PPO stacks at the same N. Every stack fills all 196,608 CPU replay slots from actual collected snapshots, runs 5 warmup plus 20 measured full rollout/update iterations, performs full checkpoint save/restore, and records actual GPU/host resource telemetry. PPO releases unused dual teachers before its phase. No probe state enters formal learning. Initial fill is an additional explicitly billed 48-step rollout. GPU own-process telemetry must be present; missing PID data is not treated as zero usage. Thresholds are total GPU≤7680 MiB, own GPU≤7168 MiB, host available≥4 GiB, no nonfinite/capacity/OOM error. Probe result/selection bind SDS and the exact code fingerprint; nonempty reused output directories are refused.

Export requires an accepted candidate checkpoint and emits a single traced Actor with normalization/history/map/route-input contract, without teachers.

## Verification

- `python -m pytest tests/test_v10_learning_core.py tests/test_v10_runtime_core.py -q`: 23 passed, one upstream Torch JIT deprecation warning (latest run; see parent integration output for final broad suite).
- Actual CUDA RNG before/after independent Critic initialization is bitwise equal; CPU initialization uses the CPU default generator only.
- Actual six-environment, 48-control-step smoke: **288 executed engineering samples**, all six roles, finite snapshots/labels/rewards. Evidence: `logs/a2_v10/runtime_smoke_20261126/smoke.json` and its append-only cost journal.
- Real D1 and D3 one-environment assemblies at zero control steps produced the same robot physics hash `aa74d2d53526f2025b46054771ec1cf9e785b9dcee7a860e98e2c8303ee58442` with distinct terrain hashes. Evidence: `logs/a2_v10/physics_course_hashes.json` and `.log`.
- Unit cases include real MuJoCo CPU models with different terrain inferred masses, robot mass-change detection, no GPU PID admission, old-checkpoint external-cost monotonicity, reservation crash/duplicate resume/idempotency/boundaries, independent update-cost accounting, exact integer metadata, full PPO+keep microbatch gradient equivalence, and partial-rollout shifted checkpoint thresholds.

## Not verified by this agent

Complete 1024/512 resource probe has not been launched pending independent review and parent scheduling. Teacher behavior admission, formal DAgger/PPO training, export deployment playback, and final behavior acceptance remain unverified. No improvement is claimed. Two failed zero-step diagnostic invocations (local `/tmp/profile.py` import shadow and initial actuator-index attribute naming) consumed zero physics samples and were corrected. The smoke predates expanded robot-only physics hashing; its physics hash is historical, while the cross-course hash evidence uses the final filter.

## Q4 correction and retest — 2026-11-26

此前工程审核遗漏：正式 `worker` 的周期保存分支仍引用未定义的 `last_save_bucket`，先前只有 `save_due` 单函数测试，未覆盖完整worker首次保存。此前的“运行时完成”不代表该路径已验证。现已修回 `save_due(last_saved_confirmed, consumed)`，并通过真实CPU worker/model/Adam/checkpoint seam测试：维持原始2,457,600周期、512×48样本单位，运行到100轮首次保存，完整恢复后到200轮跨下一阈值；周期与结束重合时用同一磁盘inode的边界名称复用一次序列化，实际文件SHA一致。虚拟collector仅用于持久化工程测试，不冒充物理训练。

Q4只改内存生命周期，不改配方、容量、样本或资源门槛：

- SHA统一8MiB流式；保存完成后全文件SHA与ZIP CRC验证。旧replay、无用rollout和序列化临时视图释放后，才实际load。load失败保留源checkpoint。
- 正式worker和probe共用单所有权replay移交：`take_state_dict`消费解码state中的storage，源mapping清空，防止可变别名残留；model、Adam和RNG移交后释放旧解码容器。每阶段记录RSS、进程峰值RSS和MemAvailable，连续ResourceMonitor峰值不重置。
- collector按同一time-major/env-major顺序预分配完整CPU FP32缓存并同步逐帧复制，避免48帧列表与concat结果同时常驻。replay逻辑采样组合原indices，仅复制选中256条，保留完全相同RNG消耗、新旧比例和有放回行为。带梯度源逐slot detach，入库不挂计算图。
- probe恢复核验完整replay内容/metadata/int64/count/seen/RNG、下一批采样及模型/Adam/global RNG；比较下一批时保存并恢复sampler RNG，不污染下一次真实采样。
- 完整CPU测试（显式 `V10_FULL_REPLAY_CPU=1 ...pytest tests/test_v10_runtime_memory.py -q`）：**8 passed, 69.18s**。完整196608条、所有45/5×45/3×40×30/3/35观测和12标签均FP32；实际checkpoint **3,100,924,643 bytes**；RSS满replay4.136GB→释放后1.035GB→load后4.153GB，峰值4.214GB，阶段MemAvailable最低约9.319GB。证据 `logs/a2_v10/q4_cpu_full_replay_memory.json`、`q4_cpu_memory_test.log`。

本次Q4没有GPU上下文、没有新增物理控制样本、没有运行新probe。既有737280条probe费用不删除、不退款。CPU内存等价通过不代表GPU双栈资源准入或行为验收通过；重新probe仍须显式全局费用授权、独立审核与主代理调度。
