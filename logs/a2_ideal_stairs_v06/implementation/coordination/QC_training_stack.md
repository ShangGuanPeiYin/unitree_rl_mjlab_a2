# C 训练栈（model/PPO/runner/评估/入口）Q001/Q002 实施记录

- 作者：执行子 agent C；日期：2026-11-16。
- 合同：SDS v0.6 + `Q001_v06_decision.md`（SHA `5f3f2aa3…a4b1`）+ `Q002_v06_decision.md`（SHA `3ebc7441…ccd52`）+ `W0_FREEZE.md`/`W0_FREEZE_ADDENDUM_1.md`。
- 本文件不修改任何数值；不代表工程验收或训练效果通过。未 commit/push，未启动 GPU/训练/仿真。

## 0. ROOT 解析回归（集成检查发现的阻断缺陷）

- 修复：`src/tasks/velocity/v06/orchestration.py::ROOT` 由 `parents[3]`（=`src`）改为 `parents[4]`（项目根，与 `spec.ROOT` 一致）。
- 同批核对：`_rooted()` 让 `--output-dir`、`--acceptance-file`、磁盘守卫、`preflight_dir`、worker registry 相对路径一律以项目根解析；`validate_arguments(root=...)` 可由 preflight/orchestrator 注入根；`scripts/{run,train,evaluate,play,smoke,test,check}_a2_ideal_stairs_v06.py` 均为 `scripts/` 下的 `parents[1]`（项目根），并新增 `_rooted()` 统一解析 run 内相对路径。
- 回归证据：`tests/test_v06_orchestration.py::test_root_is_the_project_root_not_src`、`::test_preflight_writes_under_the_project_root_not_src`（临时 root 下 preflight 落 `<root>/logs/...`，不产生 `<root>/src/logs`）、`::test_existing_output_directory_is_refused_and_untouched`（拒绝且不清空）。

## 0b. SDS 9.3 资源与恢复 probe

- 入口：`run_a2_ideal_stairs_v06.py --mode probe [--execute]`；无 `--execute` 只写 `preflight/preflight_probe_seed42.json`（不建 probe 目录、不初始化 CUDA）；probe 目录固定 `<root>/logs/a2_ideal_stairs_v06/implementation/probe/probe_seed{seed}`（拒绝已存在、拒绝复用 train run 目录），带 `--resume-run` 被拒绝。
- 成本（`orchestration.probe_plan()` 断言）：512 训练环境 + 8 评估环境、最终 L40(21) 地形、5 预热 + 50 测量 + 1 恢复 = **56 学习更新**；**1,376,256 控制样本**、5,505,024 物理子步、1,120 Adam；`counts_against_training=False`、`training_budget_untouched=30000`、`training_budget_consumed=0`。
- 资源：50 次测量窗口内逐次采样整卡/进程/host 可用/磁盘，输出 min/max/median 与四项阈值判定；不足 → `state=resource_insufficient` 并保留阻断项，不自动减环境或 rollout。
- 恢复：第 55 次后写 schema8 checkpoint（容器标签 `v06-probe-state-v1`），`--resume-run` 语义由 probe worker 内部验证（`recovery.preload_checkpoint(mode='probe')` + `resume_contract(allow_probe=True)`）：`resume_mode=new_episode`、`resumed_from_update=55`、`updates_after=1`、`billed_updates_not_replayed=True`、`initialized_from_eth_again=False`；正式训练 `mode='resume'` 拒绝该容器标签。`ProbeLedger` 拒绝重复计费与二次 resume。
- 产物：`probe/probe.json`（plan/cost/phase_timeline/resources(min/max/median+阈值)/recovery/checkpoint SHA/`acceptance_scope=engineering_check_only`/`resumable=false`/`probe_sha256`）+ `probe/preflight.json`；失败时 `probe/error.json`。
- 证据测试：`tests/test_v06_orchestration.py::test_probe_plan_cost_identity_and_budget_isolation`、`::test_probe_without_execute_has_no_side_effects`、`::test_probe_execute_writes_engineering_evidence`、`::test_probe_resource_insufficient_path`、`::test_probe_recovery_validation_rejects_replays_and_wrong_containers`、`::test_probe_ledger_records_56_updates_and_one_recovery`；容器拒绝：`tests/test_v06_recovery.py::test_probe_checkpoint_is_refused_by_formal_resume`、`::test_build_checkpoint_can_emit_the_probe_container`。
- probe 不跑 gate/diagnostic/retention 评估（`V06Runner(evaluate=False, autosave=False)` 由 probe worker 使用），因此不依赖 B 尚未提供的 `stall_samples`。

## 1. Q001-7：停滞 collapse 常量与判定

- `speed.diagnostic_collapse_streak()` / `require_diagnostic_collapse_streak()`：只读 `spec.DIAGNOSTIC_COLLAPSE_STREAK`，**无静默 fallback**；缺失、非整数、`<2` 或不在 `spec.reward_config()`/`spec.contract()` 中时报 `speed.MissingConstant`。`orchestration.validate_arguments`（预检）调用该函数，预检记录写入 `contract.diagnostic_collapse_streak`（value/checks/config SHA）。
- 其余继承 r2 停滞参数（0.80 / 12 / 12 / 100）只从 v0.6 spec（若提供）或只读 `v05_r2/runner.py` 继承常量读取，不在 C 侧写死。
- `evaluation.diagnostic_decision(records, phase, generation_updates, generation)`：15 例（每档 5，速度组取当前宏阶段：早期 .3/.5/.8，V/R .2/.4/.6）；`stall_samples/active_samples≥0.80` 数量 ≥12 与楼梯“无有效首阶”≥12 **各自独立**（不是交集），平地无首阶条件；当前 generation ≥100 更新才合格；缺例/零分母/缺字段 → `EvaluationError`（不是不命中）。
- `evaluation.diagnostic_collapse_state`：新 generation 清零、有效不命中清零、连续命中累计到常量 2 → `collapse`；runner 映射为 `behavior_collapse`。该 2 只用于此计数，与 flat/H00 保持集、最低子级 4 窗口计数（均在 B 的课程状态）完全分离。
- 证据测试：`tests/test_v06_evaluation.py::test_diagnostic_*`、`tests/test_v06_entrypoints.py::test_collapse_streak_is_required_without_a_fallback`、`tests/test_v06_runner.py::test_diagnostic_*`。

## 2. Q002-1：F0→H 连续两次门禁

- 规则由 B 的 `CurriculumManager` 执行；C 侧每个完整更新边界在 tick 前评估当期门禁，并以 `on_update(update, gates=[gate])` 交给课程，课程在 post-tick 计数、提交合法晋级、最后判预算。
- C 侧只保留门禁证据与 `pending_transition`（不驱动晋级）。`PHASE_GATE_RULES` 仅用于证据字段（required=2，V/R 有 streak 起点与最小更新），与 spec/B 一致。
- 证据测试：`tests/test_v06_runner.py::test_f0_passes_at_100_and_200_advance_to_h`、`..._second_gate_failure_blocks_until_a_later_pair`、`..._last_update_legal_second_pass_advances_before_budget`、`..._budget_exhausted_without_a_second_pass_is_gate_failed`（均用真实 `CurriculumManager`，无仿真）。

## 3. Q002-3：最终冻结测试

- `evaluation.spawn_profile(phase, kind)`：`kind='test'` 显式生成 `full_R_expanded`（`x∈[-1.20,-0.80]`、`y±.10`、`yaw±10°`、独立均匀、无训练 ramp），并写入 case 表的 `spawn_profile`/`spawn_parameters`/`target_tile_id=21`/`target_lesson='L40'`，每例 `tile_id=21`；`check_table` 强制这些字段，并用同一生成器**重新生成** case 表逐值对比（改表/换标签会被拒绝）。
- 生成器证据测试：`tests/test_v06_evaluation.py::test_final_test_table_uses_the_full_R_expansion`（按标记参数重构全部抽样）、`::test_final_test_generator_receives_the_declared_bounds`（spy 记录 `uniform` 实参 = 三组扩张范围）、`::test_a_test_table_with_the_narrow_profile_is_rejected`。
- 流程：R 两次门禁 → `lifecycle_state='ready_for_final_test'`（不是训练通过）→ 300 例每档 ≥95 → `complete`；有效未达标 → `gate_failed` + `stop_reason='final_test_failed'`，保留各档 `case_pass_count`；评估异常/缺例 → 保留 `error_type`，不记有效失败、不 complete（`tests/test_v06_runner.py::test_final_test_failure_keeps_per_speed_counts`、`::test_final_test_engineering_error_is_never_complete`）。
- 总预算恰为 30000 且 R 已合格：允许零学习最终测试（`::test_total_budget_exactly_ready_allows_the_zero_learning_final_test`，`ppo.collect_calls==0`）。

## 4. Q002-4：状态源与终态证据

- `runner.STOP_STATES`/`LIFECYCLE_STATES` 直接取 `spec.STOP_STATES`/`spec.LIFECYCLE_STATES`；`recovery._validate_ledger` 同样校验八类停止态与三生命周期态，`complete` 必须带 `full_R_expanded`、每档 100 例且 `passed=true` 的最终测试证据；停止态 run 与 complete run 都拒绝 resume（`resume_contract`）。
- `pilot_not_applicable`/pilot 映射已从主路径删除；`result()`/`stop.json`/`terminal.json`/checkpoint ledger 统一携带 `state/lifecycle_state/stop_state/reason/error_type/global_update/phase/substage/generation/budget(总/阶段/子级)/evidence/final_test`。
- 工程错误保留 `error_type`：runner 异常路径写 `exception.json`；`orchestration.execute()` 写 `control.json`（`action=failed`、`error_type`）与 `error.json`，`MonitoringBlocked` 映射为 `monitoring_blocked`。
- `orchestration.request_stop` 只接受 spec 停止态名（用户停止 `failure_class=user`，监控故障 `communication`）。
- 证据测试：`tests/test_v06_runner.py::test_terminal_evidence_is_complete_and_uses_the_spec_states`、`tests/test_v06_recovery.py::test_stopped_checkpoint_is_refused_for_resume`、`::test_unknown_state_names_are_foreign`、`::test_complete_requires_the_passed_full_final_test`、`::test_complete_with_a_training_stop_is_rejected`、`::test_gate_failed_requires_a_stop_reason`、`tests/test_v06_orchestration.py::test_stop_reasons_come_from_the_frozen_spec`、`::test_worker_engineering_error_is_typed_and_never_complete`、`::test_monitoring_failure_is_a_monitoring_blocked_stop`。

## 5. Q002-5：预算判定顺序与同时发生

- 边界顺序：门禁/诊断/保持集先评估（门禁必须对刚训练完的更新评估）→ `on_update` 内先 ingest 门禁、提交合法晋级、最后判预算 → 保存 interval/gate/transition 证据 → 环境 schedule → 终态/预算分类。
- 优先级（实现于 `_check_stop_conditions` + `_consume_terminal_event`）：
  1. `complete`/`ready_for_final_test` 生命周期优先于任何预算标签；
  2. 行为/课程停机（`behavior_regression`/`behavior_collapse`/`curriculum_stalled`）不被预算覆盖；
  3. `monitoring_blocked`、`user_requested_stop`、`resource_insufficient` 阻止继续评估；
  4. 总预算耗尽 → 主状态 `budget_exhausted`，证据同时记录总/阶段/子级耗尽；
  5. 未耗总预算而阶段/子级耗尽且课程 `finished` → `gate_failed`，`reason=phase_budget_exhausted`/`substage_budget_exhausted`；
  6. 单次门禁失败但预算有余额 → 不停机；阶段最后更新合法第二通过 → 先晋级。
- 证据测试（同时发生，而非孤立布尔）：`tests/test_v06_runner.py::test_total_and_phase_budget_in_the_same_frame_prefers_budget_exhausted`、`::test_substage_budget_exhaustion_reports_its_scope`、`::test_a_single_failed_gate_with_budget_left_does_not_stop`、`::test_behavior_regression_is_never_overwritten_by_budget`、`::test_total_budget_after_a_mid_phase_advance_is_still_budget_exhausted`。

## 6. Q002-6：保持集

- 消费 B 的 `retention_due(update) -> tuple[str,...]|None`（`RETENTION_KINDS=('flat','h00')`，`RETENTION_GATE_KINDS` 映射），不再使用 `str(due)` 单值；每个 kind 各跑 15 例（每档 5）并调用 `record_retention(kind, failed_cases, passed, update=...)`；每集合独立 streak/清零，`behavior_regression` 由 runner 执行；checkpoint 保存 B 的 `retention_qualified_at/next_due/last_completed/streaks/enabled` 镜像。
- Pending 评估事务：执行前写 `pending_evaluations.json`+ledger，完成即清除；恢复时 `_complete_pending_evaluations()` 按原 update 补完一次，B 的 `record_retention` 按 update 幂等，不重复累计。
- 证据测试：`tests/test_v06_runner.py::test_retention_due_returns_all_sets_and_runs_each_once`、`::test_retention_pending_evaluation_completes_once`、`::test_one_retention_kind_never_clears_or_accumulates_the_other`、`::test_retention_failure_twice_is_behavior_regression`。

## 7. Q002-7：V ramp 与恢复

- runner 每个完整更新边界调用 `env.set_schedule(schedule, stage_iteration)`，`stage_iteration` = 当前阶段内已完成 PPO 更新数（B 的 `spent_by_phase[phase]`），不是全局 update；进入 V 时为 0，reset/eval/checkpoint/resume 不清零；episode 命令采样由 B/env 在 reset 时执行。
- 证据测试：`tests/test_v06_runner.py::test_v_ramp_uses_the_in_phase_update_count_and_survives_resume`（u=0/1/500/1000 传入值；checkpoint→resume 后仍为 500，下一次更新 501）。

## 8. 接线与其他

- `scripts/train_a2_ideal_stairs_v06.py::build_telemetry`：`A2_V06_WANDB`/`WANDB_API_KEY` 启用时用 `connection.ReliableConnection(directory=output_dir/'telemetry', backend=WandbBackend(...))`，否则同一接口的 `V06Telemetry`；训练结束统一 `finish(result)`；传输策略 2s/30s/120s/32 在飞未改。证据：`tests/test_v06_entrypoints.py::test_train_script_uses_the_reliable_connection_when_wandb_is_enabled`。
- `scripts/run_a2_ideal_stairs_v06.py`/supervisor 的 `worker_factory` 仍由主对话在 GPU 运行时接线（C 未启动 GPU）。
- 诊断 case 证据字段：`environment_case_runner` 从 episode 记录取 `control_samples`(→active_samples)、`first_step_acquired`(→first_step_valid)，`stall_samples` 从 episode/info/diagnostics 读取。**当前 `v06/env.py` 尚未输出 `stall_samples`**（也无 stall 检测），因此真实诊断会在 `diagnostic_decision` 报评估错误——按 Q001-7 这是正确失败模式，但需要 B/主对话补 env 的停滞计数接线；C 侧不改 env。

## 0c. Q003：F0 净位移口径（evaluator/runner 侧）

- `speed.forward_distance_definition()`：优先读 `spec.FORWARD_DISTANCE_DEFINITION`/`forward_distance_definition`，缺失时使用 Q003 冻结字面 `net_displacement_pre_reset_500_steps`（在 `CURRICULUM_SETTINGS`/报告中标明来源；A 尚未把该常量写进 spec 时不会静默改口径）。
- `speed.net_displacement(start_x, end_x)` = 有符号 `x_local[500,pre_reset] - x_local[0,after_reset]`；`speed.positive_path_length()` 只作诊断；`flat_case_metrics(forward=..., start_x=..., end_x=...)` 要求有符号净位移，三者同时给出必须完全一致；`flat_case_pass` 判据 `D_net >= 0.5*c*10`（`>= required - 1e-9`），不 clamp、不取绝对值；c=.3/.5/.8 门槛 1.5/2.5/4.0，其他 c 走通式。
- `runner.environment_case_runner`：F0 只从 episode 的**冻结 pre-reset 快照**取起点/终点（`start_local_x/end_local_x`、`local_x_start/end`、`x_start/x_end`、`x0/x500`、`flat_start_x/end_x`）或 env 的净位移字段（`net_displacement`/`flat_net_displacement`/`forward_distance`）；两者同时存在且相差 >1e-9 报评估错误；无快照/净位移、无 y/yaw 证据、或“无 failure 却不足 500 步”均报评估错误（不通过、不缩短分母）。真实 499 步提前 failure 保留自身终止快照并计失败。
- `evaluation.summarize`：flat case 只有在有 Q003 证据或显式 verdict 时才判定；metrics 与 `flat_pass` 不一致报错；`CaseTable.to_json`、`summarize`、`gate_result`、`runner.gate_identity` 均记录 `forward_distance_definition`（进入 case 表/门禁身份 SHA）。楼梯门禁不新增距离条件。
- 证据测试（`tests/test_v06_evaluation.py`）：Q003 六类表（恰好通过/max 高净不足/往返净 0/先退后进净够/净后退为负/差 1e-6）、起点 -0.9/-1.0/-1.1 与整 tile 世界平移不变、c=.5/.8 通式、距离够但 y/yaw/failure/499 步不合格、start/end 与 forward 不一致、flat batch 一败一过（19/20 仍过 18/20 门禁但按签名逐例判定）、不完整 flat 评估报错、env 快照读取 500/499/提前 failure 三例，以及 `forward_distance_definition` 出现在 case 表/汇总/门禁身份。

## 0d. 真实 env 集成与 probe 恢复 allow_probe

- `runner.environment_case_runner` 新增 `_case_scalar/_case_field`：B 的 `info`/`diagnostics` 为 per-env `[B]` 张量时，按“逐元素必须一致”归约为标量（不一致 → `EvaluationError`，不取 env0）；优先使用 `info['episodes'][-1]` 与 `env.diagnostics_cases()[0]` 的逐 case 标量，并与 per-env 张量交叉校验；`forward_distance_start/end`、`forward_distance_terminal`、`max_lateral/max_abs_y`、`max_abs_yaw_deg`、`active_samples`、`stall_samples`、`first_step_valid/first_step_acquired` 均走同一读取。
- 真实 CPU 集成测试（`tests/test_v06_runner.py`）：`V06Env(num_envs=8, evaluation=True, fixed_vx=0.3, initial_cases=[case], frontier_tile_id=0)` + `environment_case_runner` 跑 1 个 F0 case 500 步 → 返回标量 `steps=500/forward_distance/max_abs_y/max_abs_yaw_deg/active_samples/stall_samples/first_step_valid/flat_pass`，并断言 `env.diagnostics_cases()` 的 8 个 env 值一致；`step_limit=499` 且无 failure → `EvaluationError`（不完整评估）。
- `V06Runner.load(..., allow_probe=False)`：`allow_probe=True`（或 runner `mode='probe'`）才允许 `v06-probe-state-v1`，并把该参数透传 `recovery.preload_checkpoint(mode='probe')` 与 `resume_contract(allow_probe=True)`；正式训练默认 `False`，probe 容器继续抛容器错误。`build_probe_worker` 恢复调用 `resumed.load(..., allow_probe=True)`。
- 训练准入阶段语义：`V06Orchestrator.execute()`（train）在 CUDA 初始化前用 `stage='pre_run'` 检查（只强制整卡/host/磁盘；`process_mib` 豁免并记 `method='not_initialized'`/`exempted=['process_mib']`，证据写 `run/resources/pre_run.json`）；worker 在**第 1 个完整更新边界**用 `build_measured_resource_guard` 采样一次 `stage='measured'`，`process_mib=None` 或 >5120 MiB → runner 以 `resource_insufficient` 停机并写 `run/resources/measured_first_update.json`、`stop.json`、`result.json`；`status` 传播 worker 的真实终态（不再一律 finished）。证据：`tests/test_v06_orchestration.py::test_train_admission_pre_run_exempts_the_uninitialized_process_memory`、`::test_train_admission_still_enforces_whole_card_host_and_disk`、`::test_worker_measured_guard_blocks_when_process_memory_is_missing`、`::test_worker_measured_guard_enforces_the_5120_limit`、`::test_train_worker_wires_the_measured_resource_guard`、`tests/test_v06_runner.py::test_post_start_measured_resource_failure_stops_as_resource_insufficient`、`::test_post_start_measured_resource_pass_keeps_training`。probe 语义不变。
- 资源阶段语义（pre-run vs measured）：`resource_snapshot(..., stage='pre_run')` 在 CUDA 未初始化时把缺失进程内存诚实标为 `process_mib_method='not_initialized'`；`check_resources(..., stage='pre_run')` 只强制整卡/host/磁盘，`process_mib` 缺失时豁免并写入 `exempted=['process_mib']`（`probe.json.resources.pre_run_check.exempted`）；**测量窗口与恢复后仍 fail-closed**（`process_mib=None` → blocking，≤5120 MiB 阈值不变）。测试：`tests/test_v06_orchestration.py::test_pre_run_allows_uninitialized_process_but_measured_does_not`、`::test_measured_process_memory_still_obeys_the_5120_limit`、`::test_probe_pre_run_exemption_still_runs_the_probe`、`::test_probe_measured_process_none_is_resource_insufficient`。
- 资源进程内存（probe process_mib）：`resource_snapshot` 同时解析 `nvidia-smi --query-compute-apps=pid,used_memory`，按当前进程树（`/proc/<pid>/.../children`）匹配 → `process_mib_method='compute_apps'`；匹配不到时回退 `torch.cuda.max_memory_reserved()`（不在预检初始化 CUDA）→ `'torch_reserved'`；两者都不可用 → `process_mib=None` + `method=None` 且资源守卫 fail-closed 阻断（不再是静默 0/None 通过）。`probe.json` 写 `status`（成功 `engineering_check_only` / `resource_insufficient` / `error`）与 `process_mib_method`，`resources.measured.metrics.process_mib.method` 保留来源（mixed 标记混合）。证据：`tests/test_v06_orchestration.py::test_resource_snapshot_matches_this_process_compute_apps`、`::test_resource_snapshot_falls_back_to_torch_reserved`、`::test_nvidia_smi_parses_gpu_and_compute_apps_tables`、`::test_resource_summary_keeps_the_process_method`，probe 成功/不足路径断言 status 与 method。
- probe 累计计数：`build_probe_worker` 用 checkpoint 的 `state['learning']`（第 55 次累计）+ 恢复后 1 次更新构造 `counts`（`learning_updates=56`、1,376,256 控制样本、5,505,024 子步、1,120 Adam、`counts_against_training=False`、`training_budget_untouched=30000`），并断言单次恢复增量恰为 24,576 / 98,304 / 20；累计或增量不符、缺累计字段均抛 `ProbeError`（`probe_counts()`，回执含 `counts_at_checkpoint`/`single_update_delta`）。测试：`tests/test_v06_orchestration.py::test_probe_counts_combine_checkpoint_and_recovery`、`::test_probe_counts_reject_bad_increments_and_missing_fields`、`::test_probe_worker_uses_checkpoint_cumulative_counts`。
- 证据测试：`tests/test_v06_runner.py::test_probe_container_resume_requires_allow_probe`（allow_probe 恢复 55→56、默认 load 拒绝、`resume_contract(allow_probe=False)` 抛 `ForeignContainerError`）、`::test_probe_worker_source_uses_allow_probe_and_keeps_the_cost_receipt`、`::test_real_env_case_runner_returns_consistent_scalars`、`::test_real_env_499_steps_is_an_incomplete_evaluation`。

## 9. 测试

- C 9 文件：`233 passed, 0 failed`（含 2 条真实 CPU V06Env 集成用例）。
- 全量 launcher（含 A/B）：`404 collected / 404 passed / 0 failed / 0 skipped / 0 error`，exit code 0。

### Q004（SHA 768dc38a…8efe07）实施
- A（快照）：`evaluation._optimizer_digest` 覆盖 `param_id -> state` 映射（`.items()`）、step/exp_avg/exp_avg_sq 及任意非 tensor 结构、param_groups 的 lr/betas/eps/weight_decay/params 身份；数值 token 化（NaN/Inf 显式 token，不吞）；读取只读且不保留 tensor。非空真实 Adam（1 次/多次 step、空 state、分别改 exp_avg/exp_avg_sq/step/lr/param_id/额外字段、以及真实 `evaluate_checkpoint` 路径）已测。
- B（两阶段）：`V06Runner.commit_learning_update(u)` 只把成本计入旧 phase/substage/generation（`sum(spent)==learning.update`，重复幂等、倒退拒绝），不消费窗口/不晋退/不判预算；`finalize_boundary(u)` 在旧阶段执行到期门禁/诊断/保持集后，撤销一次 tick 并调用 B 的 `on_update` 完成一次窗口/晋退/预算/完成判定；`pending_boundary` 记录 update/旧 phase/substage/generation/到期任务与 case 表 SHA/结果 ID/是否已决策；结果按 `(update,kind,retention_kind,generation,case_table_sha256)` 去重；恢复时 `_complete_pending_boundary()` 只补未完成评估与决策，不重做 PPO、不重复计费。
- C（持久化）：worker 在任何基线评估前保存 `update0`；每个到期评估边界在评估前保存带 `pending_boundary` 的完整 checkpoint（`boundary_<u>.pt`，原 100 例行保存与阶段转换保存保留）；异常时保留原始错误与最近有效 checkpoint，`save('exception')` 带 pending_boundary，保存再失败另写 `checkpoint_save_error`/`incomplete_save_error` 且不覆盖原异常；PPO 半途失败写 `incomplete_update`（实际 Adam 计数、不可续接），recovery 拒绝恢复。
- D（故障注入）：update 50 诊断抛错 → checkpoint 含 learning.update=50/spent=50/1,000 Adam/1,228,800 控制样本/pending 诊断，正式 `validate_checkpoint` 通过且 `resume_contract.pending_boundary_requires_completion=true`；恢复后只补诊断与决策、不重训/不重复计费，下一次学习为 51。测试：`tests/test_v06_runner.py::test_pending_diagnostic_resumes_without_rebilling_and_next_update_is_51`、`::test_exception_boundary_writes_a_pending_update_checkpoint`。
- E（probe）：probe 现以 `evaluate=True` 走真实 runner（update 50 真实 15 例诊断 + 真实非空 Adam 快照 + `boundary_000050.pt` 保存），仍 55 保存→allow_probe 恢复→56 补 1；资源四项/工程容器不可用于 trial02 不变。
- 未改动奖励/PPO/课程/门禁/速度/seed；`train_seed42_trial01` 未改动。

### 训练 update 50 崩溃修复（运行期 bug）

- Bug 1：`evaluation.training_state_snapshot` 对 `optimizer.state_dict()['state']`（param_id -> dict）错误使用 `enumerate()`；改为 `.items()`（键统一 str），非 mapping 条目显式 `EvaluationError`。测试 `tests/test_v06_evaluation.py::test_optimizer_snapshot_iterates_the_state_mapping`（一步 Adam 可快照、再一步被检测为 changed）。
- Bug 2：异常路径保存失败（learning.update 比 curriculum spend 多 1）。`V06Runner.save(..., pending_update=)` / `checkpoint_state(pending_update=)` 显式携带“一个已计费未提交更新”；`recovery.build_checkpoint(pending_update=)` 写入 `ledger.pending_update`，`validate_curriculum/audit_budget` 仅在该标记下允许差值恰为 1；异常路径 `run()` 用 `save('exception', pending_update=self._pending_update_marker())`，保存失败再写 `exception_save_error.json`，`exception.json` 始终存在。pending checkpoint 是证据，`resume_contract` 明确拒绝其恢复。
- 测试：`tests/test_v06_runner.py::test_diagnostic_boundary_saves_with_committed_curriculum_spend`（update 50 保存、spend==learning.update）、`::test_gate_boundary_saves_with_committed_curriculum_spend`（update 100 interval 保存、spend==100）、`::test_exception_boundary_writes_a_pending_update_checkpoint`（诊断抛错仍写出 `exception_000050.pt`、pending=50、spend=49、可校验、resume 被拒）。
- 未改动任何 logs/训练目录；`train_seed42_trial01` 保留原状。
- probe 入口回归：`main(..., probe_factory=...)`/`V06Orchestrator(..., probe_factory=...)` 已接线，缺失 probe worker 时明确 `AdmissionError` 而非回退训练 worker；本地实跑 `--mode probe --help` exit 0、`--mode probe`（无 execute）exit 0 且只写 `preflight/preflight_probe_seed42.json`、不建 probe 目录、无 `src/logs`（`tests/test_v06_orchestration.py::test_main_accepts_a_probe_factory_and_help_exits_zero`、`::test_main_probe_without_execute_only_preflights`、`::test_main_probe_execute_requires_a_probe_worker`、`::test_run_script_wires_the_probe_factory`）。
- smoke 账本回归：`history[]` 记录每次更新的真实增量（`update_cost_deltas`：384/1536/20 每更新），两次合计 768/3072/40 并在写入前断言（`tests/test_v06_entrypoints.py::test_smoke_history_records_real_per_update_deltas`、`::test_smoke_history_fields_are_not_episode_counts`）。
- 二者均不代表工程验收或训练效果通过。

## 10. 遗留（非数值裁决项）

1. 需要 B/env 提供逐 case 的 `stall_samples`（见 §8），否则真实诊断无法产生 collapse 证据。
2. 主对话需在 GPU worker 中接 `build_worker` 的 `worker_factory` 与 `ReliableConnection.start()`（当前 CPU 测试用 fake backend）。
3. `spec.recipe_config` 仍把 `output_dir`/`acceptance_file` 作为可断言字段；orchestration 将二者视为运行路径（`RUN_LOCAL_FLAGS`），Q001-5 的固定输出/验收路径由 C 侧 `validate_arguments` 在 `--execute` 时校验。若 Codex 要求改由 spec 统一断言，需 A 调整。
