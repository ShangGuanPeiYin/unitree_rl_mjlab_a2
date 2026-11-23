# S2.1 / DSH-Q036–Q037 实现记录（质量判据、top 安全登顶、时间语义）

- 仓库：`/home/xxs/dog/unitree_rl_mjlab_a2`，分支 `main`。
- 依据：`doc/sds/reviews/v09_execution_rulings.md` DSH-Q035（场景）、DSH-Q036 A/B（质量/top/时间）、DSH-Q037（c_request、奖励耦合、pitch 窗、单一来源）、SDS v0.9 §4.2/§5.2/§5.3/§5.4/§6.2/§6.5。
- 本文件只记录实现与实测，不代替 SDS 裁定。**本次未提交、未 push**（由 lead 统一提交）。

## 1. 文件与改动位置

| 文件 | 改动 |
|---|---|
| `src/tasks/velocity/v09/s2_quality.py`（新） | 共享的 per-env 质量状态机、离线 `evaluate_frames`、gate 使用的 `acceptance_reasons`。 |
| `src/tasks/velocity/v09/s2_lifecycle.py` | 计划终点 1000/1400/1500；400 cruise + ramp k=1..100；按 Actor **实际消费**的 `c_exec` 锚定 first-zero/HOLD；`c_request` 单一开关；输出 `*_at_actor`/`*_after_step`/`ramp_k`；`quality_reasons` 委托共享核心。 |
| `src/tasks/velocity/v09/s2_train_real.py` | `_s2_task_state` 接入质量机（重置/更新/显式成功或质量失败）；修正 top 捕获；`_s2_reward` 只消费 lifecycle 的 `*_at_actor` 状态；删除重复 stop 状态机；`stop_mode` 写入 plan row。 |
| `src/tasks/velocity/v09/s2_physics.py` | `_s2_patch_context` 在 v07 success 被清零前捕获 raw safe-summit；`top_arrived`/`v07_success` 改用该信号；fake backend 同步为 front+0.8 安全登顶。 |
| `src/tasks/velocity/v09/s2_acceptance.py` | `quality_reasons` 委托 `s2_quality.acceptance_reasons`（T-3；这是本任务对原 ownership 清单的扩展，理由见 §7）。 |
| `src/tasks/velocity/v09/metrics.py` | F-3：gate/final 巡航窗改为共享窗口（flat 100..999/100..399；stairs/platform 四足入台+1 m 至 decel/终点），并输出 `quality_window_start_step/end_step`。 |
| `tests/test_v09_s2_0_metrics.py` | F-3 回归：窗外 16°/10 帧不产生 gate reason，窗内才产生。 |
| `tests/test_v09_s2_quality.py`（新） | 24 例独立参考用例 + T-1/T-2/T-6/Q037 回归。 |
| `tests/test_v09_s2_lifecycle.py` | 1000/1400/1500、400+ramp、`*_at_actor`、`c_request` 开关兼容。 |
| `docs/v09_s2_q036_implementation.md`（本文件） | 证据与说明。 |

关键行号（本快照）：
- 生命周期 `S2Lifecycle.update`：`s2_lifecycle.py:116`；消费快照 `:132`；`c_request` 开关 `:17` 与 `:166`；ramp k `:173`；实际 first-zero `:187`；端点/HOLD `:204`；消费 phase 输出 `:268`。
- 训练接入 `_s2_task_state`：`s2_train_real.py:207`；top 捕获 `:326`；质量更新 `:399`；lifecycle 调用 `:440`；`_s2_reward` `:453`；消费命令 `:525`；20 项 reward 值 `:594`。
- 质量核心：`s2_quality.py:98`（cruise metrics）、`:114`（cruise reasons）、`:131`（stand metrics）、`:155`（stand reasons）、`:177`（acceptance）、`:234`（`S2Quality`）、`:431`（`update`）、`:557`（窗内 pitch latch）、`:872`（`evaluate_frames`）。
- gate/final 委托：`s2_acceptance.py:18`；`s2_lifecycle.quality_reasons` `s2_lifecycle.py:282`。
- top 捕获：`s2_physics.py:544`（`_RawV07Success`）、`:566`（context wrapper）、`:654`（`capture.record(result.success)`）、`:880`（`take`）、`:913`（`top_arrived=top_raw`）。
- F-4 D_post zero：`s2_lifecycle.py:52,145-147`（`top_x` + `dpost`），`s2_quality.py:304,496,625-630`；gate 窗口：`metrics.py:146-195`。

## 2. `top_arrived`：原 v07 安全登顶信号

**不复用**四足 XY 入台 + 机身 `front+0.4 m` 的 §5.2 门控起点。`top_arrived` 现在来自 v07 原始 `events.evaluate` 的 `success`，在 S2 patch 清零前由 `_RawV07Success` 捕获（`s2_physics.py:544-585`, `:640-655`），训练适配器每帧 `take()` 一次并只对 climb task 生效（`s2_train_real.py:326-332`）。

冻结条件（全部来自原 v07/SDS 表，非新阈值）：
- 机身至少越过平台前缘 **0.8 m**：`SUCCESS_BODY_PLATFORM_MARGIN=0.8`（`src/tasks/velocity/v07/spec.py:645`），判定 `body_x >= x_top + 0.8`（`src/tasks/velocity/v07/events.py:243-246`）。
- 足点在平台真实边界内缩 **0.042 m**：`SAFE_INSET=.042`（`spec.py:177`），`feet_safe_platform` 用 `x0=n*d+inset`、`x1=bounds[1]-inset`（`events.py:270-279`）。
- 单足 **2 连续帧**顶面接触、当前 **≥2 足**支撑、最近 **50 帧**内支撑、**5 连续帧**确认：`spec.py:642-648`，`events.py:231-254`。
- 横向 `|y| <= 0.20 m`：`spec.py:646`，`events.py:243-245`。
- 失败优先：`success = result.success & ~failure`（`events.py:330`）。
- 真实地形/平台边界：使用 v07 `tile_parameters` 的 `N*d` 与实际 `bounds`，S2 terrain patch 只替换几何表，不替换该判定（`s2_physics._s2_patch_context`）。

该信号与 §5.2 门控分开记录：训练 trace 同时给出 `four_feet_entry_step`（四足首次同时入台）、`g_level`、`top_arrived`、`cruise_start_step`（`s2_train_real.step` info，`:700-760`；`s2_quality.result()` 同名字段）。

**F-4 / Q019 A3 / Q036：** `D_post` 的零点是 top latch 同帧的机身 `route_x`（`top_x`），不是平台前缘。`S2Lifecycle` 在 top 事件帧写 `top_x` 并用 `local_x - top_x`；`S2Quality` 同样记录 `top_x` 并用它做 stairs walk 终点与 stairs stop 请求距离。平台直启仍用 spawn+start_dist，flat 仍为时间终点。示例（前端 2.0、top=2.8858）成功点为 `x >= 7.8858`，而不是 `front+5=7.0`；测试 `test_stairs_dpost_zero_is_top_not_front`、`test_stairs_dpost_zero_is_top_not_platform_front`。S2.0/gate 缓存中若有“从前缘量距离”的字段，它们是另一个零点，不能与 `top_x` 混用。

旧 S2.0 影响：独立验证方已用原始轨迹重算 30 例（`logs/a2_v09/s2/q036_verify/s2_0_top_recompute.json`）：15/30 达到真正的安全登顶；旧报告 top 比安全登顶早 **15–19 帧（均值 16.1）**，且 15 例在安全登顶后 **25–38 帧**才物理失败。原 S2.0 的 top 窗口属于旧 proxy 版本，相关“登顶后立即失败”的时序结论应按重算结果复核；本实现产生的 `top_arrived` 与重算规则一致。

## 3. 共享质量判据（SDS §6.5 / DSH-Q036 A / Q037）

`s2_quality.S2Quality` 是唯一增量实现；`evaluate_frames` 是同一实现离线入口。gate/final 的 `s2_acceptance.quality_reasons` 与 P1-b 的 `s2_lifecycle.quality_reasons` 均委托共享阈值/比较函数（`s2_quality.py:98-231`；`s2_acceptance.py:18-26`；`s2_lifecycle.py:282-333`）。

| 判据 | 实现 |
|---|---|
| 巡航窗 | flat_walk：`2.0 <= t < 20.0`（interval 100..999，900 帧）；flat_stop：`2.0 <= t < 8.0`（100..399，300 帧）。楼梯/平台：四足入台后 `D_post >= 1.0 m` 起，含 stop-request 帧/终点。只有 `c_exec > 0` 的帧可入窗（T-1）。 |
| 空窗/<1 s | 0 帧 → `missing_window`；<50 帧 → `window_short`，永不成功。 |
| pitch RMS / P99 / speed RMSE | 闭窗后计算：RMS ≤5°，P99 ≤10°，RMSE ≤0.10 m/s。 |
| 15°/10 帧 latch | 只在巡航评价窗内累计（T-6/Q037.3），严格 `>15°`，10 连续窗内帧；出窗清零，**不跨窗累计**，不进入 HOLD/站立窗；latch 保留到终点。 |
| §6.5 stall | 仅 POST_WALK 巡航（`top_step>=0`）且 `c_exec>0.1`，沿用 51 样本/0.15·v/0.03 m/50 连续帧。 |
| 恢复 | 首个**实际消费**的 `c_exec==0` 起 150 帧内需完成 25 连续 §5.3 稳定帧，否则 latch `recovery`。 |
| HOLD | 从首个实际消费 0 起（含）计 1000 帧；flat_stop instant 终点 1400、ramp 终点 1500；到计划终点不足 1000 → `hold_incomplete`（T-2）。真实 ramp trace 在终点 hold=1001 帧（首零 500，500..1500）。 |
| 末 500 帧站立 | 均值 ≤0.05、峰值 ≤0.10 m/s；相对窗起点位移 ≤0.10 m；`|roll|,|pitch| ≤10°`；离支撑面 ≥0.30 m；四足同时 §5.3 有效支撑比例 ≥90%。 |
| 同帧优先级 | 物理失败 > 质量失败/任务超时 > 成功 > 单独 top；`failure` 抑制 +8/top +2。 |
| 输出 | 训练在终点显式给 success(+8) 或 quality failure(−10)，不再靠 hard timeout。 |

**T-1**：零命令 flat_walk 无任何 cruise 帧 → `missing_window` → −10；测试 `test_zero_command_flat_walk_must_not_succeed`。
**T-2**：HOLD 锚定实际首零；延迟 100 帧 → 计划终点 1400 时 hold=900 → `hold_incomplete`；测试 `test_hold_anchored_to_actual_first_zero_*`。
**T-6**：窗外 10 帧不 latch、窗内第 10 帧 latch、9+10 分段、跨 8 s 窗边界不累计、恰好 15° 不计数；测试 `test_pitch_over15_latch_is_window_scoped`、`test_pitch_nine_then_ten_inside_same_window`、`test_pitch_counter_does_not_span_window_boundary`、`test_pitch_exactly_15_degrees_does_not_latch`。
**T-7**：`test_all_twenty_four_independent_reference_cases_pass` 直接读取 `logs/a2_v09/s2/q036_verify/quality_cases.json` 的 24 个手算用例。flat + platform 的 21 例逐项（outcome/endpoint/window/reasons/first_zero/metrics）全部通过；3 个 stairs 参考例（`stairs_walk_pass_1m_window`、`stairs_walk_window_shorter_than_1s`、`stairs_walk_stall_latch`）的期望终点是旧的四足入台/前缘零点，在 F-4 的 `top_x` 零点下记录帧数内不会到达终点（窗口帧数与 stall latch 仍逐项核对）。这 3 例由 `test_stairs_dpost_zero_*`、`test_stairs_top_zero_short_window_is_rejected`、`test_stairs_top_zero_stall_latches_and_fails_at_endpoint` 覆盖；因此当前口径是 **21/24 逐项 + 3 例旧零点已由 Q019/Q036 取代**，不能声称 24/24。
**F-3**：`metrics.shared_quality_window` 与 `s2_quality` 使用同一窗口边界；回归 `test_f3_flat_pitch_15_latch_is_window_scoped_in_gate_metrics` 与 `test_f3_stairs_pitch_15_latch_starts_after_entry_plus_one_metre` 证明窗外 16°/10 帧不产生 gate reason、窗内才产生。

## 4. 时间/步数语义（DSH-Q036 B / Q037）

| 任务 | 计划终点（完成控制区间） | hard protection | 备注 |
|---|---:|---:|---|
| flat_walk | 1000（20 s） | 1001 | 巡航窗 900 帧 |
| flat_stop instant | 1400（28 s） | 1401 | 400 cruise + 1000 HOLD |
| flat_stop ramp_2s | 1500（30 s） | 1501 | 400 + 100 ramp + 1000 HOLD |
| stairs/platform walk | stop 请求前的距离终点 | v07 原硬时限 | 窗从四足入台+1 m 起 |
| stairs/platform stop | 实际首零 + 999 | v07 原硬时限 | HOLD 从实际首零起 |

- 行 `t=0` 是 reset 后 pre-step 样本，不计 interval；`evaluate_frames` 显式跳过（`s2_quality.py:889-895`）。真实 trace 的 `t0_rows=1`。
- ramp：`k = step - stop_step + 1`；trigger step 仍消费旧命令（reward 也按旧命令结算），下一 Actor tick 消费 k=1；k=100 为 0。`C_REQUEST_ZERO_AT_STOP_REQUEST=True`（`s2_lifecycle.py:17`）是 c_request 语义的唯一开关；Q037 已裁定“stop_request 锁存时 c_request=0 并保持到 reset，ramp 只作用于 `c_exec`”，与本实现一致。
- `c_request_at_actor` / `c_request_after_step` / `c_exec_at_actor` / `c_exec_after_step` / `ramp_k_at_actor` / `ramp_k_after_step` / `first_zero_step` / `hold_samples` 均由同一次 `S2Lifecycle.update` 输出（`s2_lifecycle.py:132-205, 255-277`）。
- **T-8 单一来源**：`_s2_task_state` 先读 v07 同帧测量（含 `env.sample_command` 实际消费命令），再更新 quality，再更新一次 lifecycle；reward 只用 lifecycle 的 `*_at_actor`（`s2_train_real.py:525-531`），`env.command` 只由 lifecycle 写一次。`phase`/`c_request`/`c_exec`/stop flags/成功失败不再有第二套计算。

真实 ramp trace（`logs/a2_v09/s2/q036/stop_ramp_final/trace.jsonl`）实测：
- step 400：`phase=APPROACH`、`c_request_at_actor=0.3`、`c_request_after_step=0`、`c_exec_at_actor=0.3`、`ramp_k_after_step=1`；
- step 401：`phase=DECEL`、`c_request_at_actor=0`、`c_exec_at_actor=0.297`、`ramp_k_at_actor=1`；
- step 499：`ramp_k_at_actor=99`、`c_exec_at_actor=0.003`；
- step 500：`ramp_k_at_actor=100`、`c_exec_at_actor=0`、`first_zero_step=500`、`hold_samples=1`；
- step 1500：`hold_samples=1001`、质量失败 −10（`recovery+stand_peak_speed+stand_displacement+stand_support`），不是 timeout。
真实 instant trace（`stop_instant_final`）：step 400 请求、401 首零、1400 终点、`hold_samples=1000`，质量失败 −10（`recovery+stand_peak_speed+stand_support`）。

## 5. 奖励耦合（Q037.2）

- 触发步（step 400）仍用触发步实际消费的 `c_request/c_exec/phase` 结算，位移/腾空/巡航权重按该步；新 step-end 请求不回写该步。
- 从下一控制步（401）起，reward 看到 `phase=DECEL/HOLD`、`c_request_at_actor=0`、`c_exec_at_actor=k` 值，`s2_rewards` 已有逻辑自动关闭 displacement/air-time、清空 air-time 累计并使用 decel 列。
- `_s2_reward` 已删除重复 stop/ramp 计算（原 `s2_train_real.py` 旧 400/1000 分支），只保留 20 项 reward math。
- 真实 trace 的 `reward_terms` 在终点包含 `feet_air_time=0`、`stable_stand=1`，trigger/next tick 的 reward 见 trace。

## 6. 实测证据（本会话真实运行）

所有命令均为单进程、顺序执行；`nvidia-smi` 显示 RTX 3070 Ti 8 GB，空闲约 6.4 GB。

```bash
# 三个角色 300 步 smoke（最终代码）
/home/xxs/anaconda3/envs/a2/bin/python scripts/smoke_a2_v09_s2_train.py \
  --course S2.1 --num-envs 7 --steps 300 --role-filter flat --execute \
  --output-dir logs/a2_v09/s2/q036/smoke_flat_300_final
# 同样 stairs / platform；结果 rows=600/900/300，completed=true

# 真实 stop/safe-top trace（14 env，1600 步；含 t=0 行）
/home/xxs/anaconda3/envs/a2/bin/python logs/a2_v09/s2/q036/run_stop_trace.py \
  --num-envs 14 --steps 1600 --mode stop_instant \
  --output-dir logs/a2_v09/s2/q036/stop_instant_final2
# stop_ramp / stairs_top / flat_all 同参数，目录分别 *_final2
```

| 证据 | 结果 |
|---|---|
| `smoke_flat_300_final/` | 2 env ×300 行，完成 |
| `smoke_stairs_300_final/` | 3 env ×300 行，完成 |
| `smoke_platform_300_final/` | 1 env ×300 行，完成 |
| `flat_all_final2/trace.jsonl` | 3×flat_walk 在 1000 步成功（reward≈8.05，含 +8）；3×flat_stop（instant 0.3/instant 0.5/ramp 0.3）分别在 1400/1400/1500 终点质量失败（reward≈−9.94，含 −10），非 timeout |
| `stop_instant_final2/` | 请求 400、首零 401、HOLD 1000、终点 1400，质量失败 −10；`t0_rows=1` |
| `stop_ramp_final2/` | 请求 400、k=1..100、首零 500、终点 1500、hold=1001，质量失败 −10；`t0_rows=1` |
| `stairs_top_final2/` | 自然策略在 step 491 触发**真实 safe-summit** `top_arrived`，`top_x=2.8529`（+2 已在该帧 reward≈2.03 中），四足入台 step 473；step 528 物理失败，入台+1 m 巡航窗未开启，故未覆盖登顶后 +8。 |
| `logs/a2_v09/s2/q036/evidence_sha256.json` | 本次 final2 证据的 per-file SHA256 与逐 env 结果/payment 摘要 |
| GPU 峰值显存（torch 统计） | 5.86 MiB（allocator peak；warp/mjlab 实际占用见 `nvidia-smi`，未另测） |

独立参考用例：`logs/a2_v09/s2/q036_verify/quality_cases.json` 的 21 个 flat/platform 例逐项通过；3 个 stairs 例的终点期望使用旧零点，已按 F-4 取代（见 §3）。旧 `v13_runB2` stop traces 在 stall 修复前崩溃（906/907）且 flat “HOLD” 实为距离触发，**不作为本实现证据**。

## 7. 未验证 / 明确缺口

- **自然策略的 stop 通过分支未覆盖**：真实 instant/ramp 都在终点质量失败（`recovery+stand_*`），没有自然通过样本；通过分支由 CPU flat/platform 21 例参考用例 + 3 个 F-4 top-zero stairs 例覆盖。
- **自然策略的 safe-summit + post walk 未覆盖**：真实 stairs 只到 safe summit + 物理失败，未覆盖 top+2 后的 +8 窗口。
- **S2.0 旧 top 窗口**：已由独立验证方用原轨迹重算（15/30 安全登顶，旧 proxy 早 15–19 帧）；本实现未重跑 S2.0 全批，只保证新信号与重算规则一致。F-4 后 stairs D_post 零点为 `top_x`；旧 S2.0/gate 缓存中从前缘量的距离不能与之混用。
- **奖励的完整 20 项在真实后端的逐项核对**：真实 trace 已记录 `reward_terms`，但独立手算 20 项在 GPU 上的逐项对照仍属 Run B/D 范围，本次未做。
- **gate/final 真实路径**：`s2_acceptance.quality_reasons` 已委托共享核心并有 CPU gate 测试（315 项总套件通过），但未跑真实 gate/final 轨迹。
- **T-5 c_request**：Q037 已裁定 stop 时 c_request 置 0 且保持；实现用 `C_REQUEST_ZERO_AT_STOP_REQUEST` 单点隔离，若后续裁定不同只需改此常量。真实 instant/ramp trace 已记录 `c_request_at_actor` / `c_request_after_step`。
- **GPU 峰值**：只记录 torch allocator 峰值，不等于整卡峰值。

## 8. 预算算术（lead 指定口径）

```text
learning_S2_1 committed        20,299,776
remaining                     779,700,224
samples per 512 rollout        24,576
complete rollouts              floor(779,700,224 / 24,576) = 31,726
31,726 × 24,576                = 779,698,176
total (committed + planned)    = 799,997,952
remaining after 31,726          = 2,048
```

## 9. 测试命令与结果

```bash
# 本任务要求的 CPU 套件（不含 gate 两文件）
/home/xxs/anaconda3/envs/a2/bin/python -m pytest \
  tests/test_v09_s2_0_*.py tests/test_v09_s2_1_train.py \
  tests/test_v09_s2_lifecycle.py tests/test_v09_s2_rewards.py \
  tests/test_v09_s2_v13_evidence.py tests/test_v09_s2_quality.py -q
# 303 passed, 1 warning

# 追加 gate/acceptance 套件
/home/xxs/anaconda3/envs/a2/bin/python -m pytest \
  tests/test_v09_s2_0_*.py tests/test_v09_s2_1_train.py \
  tests/test_v09_s2_lifecycle.py tests/test_v09_s2_rewards.py \
  tests/test_v09_s2_v13_evidence.py tests/test_v09_s2_quality.py \
  tests/test_v09_s2_gate.py tests/test_v09_s2_gate_cases.py -q
# 324 passed, 1 warning
```

## 10. 最终文件 SHA256（本快照）

```text
8433b06de6b5aa29bc0f4e9f9e3242817ed153b846545d2641c0ec3b762691ab  src/tasks/velocity/v09/s2_quality.py
ab679d1197335083d497b57d52e87fa36d0823e75cb3b57aa0426771168c3a63  src/tasks/velocity/v09/s2_lifecycle.py
de00d160b9e4930e4665e50c6420a5edfe9f6c9888abab1aa9f824821061da83  src/tasks/velocity/v09/metrics.py
5ee70704ff2d18eb5e4d400a6cbb1758d63002aaf69d49236f9900164ef0b426  src/tasks/velocity/v09/s2_train_real.py
242ca72960d70e8eef647253ce84f4e9aa12e572db13c60ac42f6e2552c0d917  src/tasks/velocity/v09/s2_physics.py
3c6b3d8574b6167a05212de6a83d0cdb0b345a04dafc317e4ca4a07f444af451  src/tasks/velocity/v09/s2_acceptance.py
8b3b25f5f615aeb44b4077a461ffcc4534d02fed4b78c6d3b1fb3f5c56962ea9  tests/test_v09_s2_quality.py
89a5ccdf0d97a617782b3d153f6019b0465256f1680c8e651b53e15c5defd4db  tests/test_v09_s2_lifecycle.py
43099a12913e333d3ceda723a518bab9faf3bf2af8deb3ed899f59bb2856e566  tests/test_v09_s2_0_metrics.py
30fbdc5c41c8594b1164a47ca28faec38f225250b6bea562ccbae1077649cfc3  logs/a2_v09/s2/q036/run_stop_trace.py
```

> 说明：`logs/a2_v09/s2/q036/evidence_sha256.json` 记录 final2 四个目录的 trace/summary per-file SHA；旧 `*_final` 目录保留未删。`tests/test_v09_s2_1_train.py` 未修改。未改动 `logs/a2_v09/s2/ledger.jsonl`、SDS 文档、`s2_train.py`、`s2_train_env.py`、`scripts/train_a2_v09_s2.py`。
