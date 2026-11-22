# A2 v0.9 S2 执行链路与 smoke 证据独立验收（V8）

- 审核 agent：A2-S2-VERIFY（独立验收子 agent，未参与实现）
- 审核时间：2026-11-22（快照 15:02–15:12）
- 仓库/分支/HEAD：`/home/xxs/dog/unitree_rl_mjlab_a2` / `main` / `2a2ae49`（`fix(s2): pass actions to backend and stop double observation updates`）
- 范围：代码两处根因修复、smoke-r9 证据复算、S1 参考对照、台账、case 选择语义；纯 CPU/只读证据，未跑物理、未改台账。
- 机器可读件：`logs/a2_v09/s2_0/verification/execution_audit_20261122_r8.json`

## 0. 结论

| 项 | 结论 | 要点 |
|---|---|---|
| 1 代码根因修复 | **通过** | 动作下发到 backend；观测只由 backend reset/step 产生一次；无二次 `update=True`/重复推进；history/map 相位正确 |
| 2 smoke-r9 证据复算 | **通过（数值）** | route +0.3783605694770813、velocity .2983198637375608、64/64 非零、contact 2.5625、g=1、无终止/失败；trace SHA 3c06e549 与文件一致 |
| 3 S1 参考对照 | **不成立（阻断）** | S1_DIAG flat case0 与 S2_DIAG flat case0 **不是同一病例**（spawn 差 0.0836 m、suite/streams 不同）；"观测等价"只能限同观测重放，不能当同病例对照 |
| 4 台账 | **部分通过（阻断 1 项）** | envelope 263,888,370、spent 384、reserved 0；10 reserve/7 commit/3 release/1 refund、op 唯一、r1 commit+refund 净 0、空尝试均释放；但 S1 parity+smoke 全记在 s2_0_matrix，满矩阵+384 会超 420,390/429,822 |
| 5 case 选择语义 | **阻断** | `S2_DIAG` 有 15 个 `case_id=0`；`--case 0` 取第一个=flat_walk 0.3，smoke 实际跑 flat；与自述"case_id0=stairs_walk"不符；无 `--task/--case-sha` |
| 6 证据 SHA 绑定 | **阻断（证据身份）** | trace SHA 3c06e549 未写入 `smoke_summary.json` 或 `00_flat_walk_real.json`（仅 npz SHA） |
| **总判定** | **代码/数值通过；台账、case、S1 parity、SHA 绑定阻断** | 不影响已记录的 384 计费，但不能据此宣称 stairs smoke 或同病例 parity |

## 1. 代码两处根因修复

- **动作下发**：`S2PhysicsEnv.step(action)` 将 `action` 传给 `backend.step(action)`；`MjlabS2Backend.step` 执行 `env.step(action)`；`_frame` 的 `a_request` 取 `action_manager.action`、`a_applied`/`pd_target` 取 `_processed_actions`。smoke trace 64/64 帧 `a_request` 非零、`a_applied` 与 `a_request` 不同，证明确实下发并经过处理。
- **观测一次**：`S2PhysicsEnv._reset_backend()` 调一次 `backend.reset()` 后以 `observation(update=False)` 缓存；`step()` 调一次 `backend.step(action)` 后以 `observation(update=False)` 重新缓存；`observe()`/`observation()` 只返回缓存。`MjlabS2Backend._frame` 不再调用 `_observations`，`step` 只用 `env.step` 返回的 obs。**未发现二次 `update=True` 或重复 `env.step`。**
- **history FIFO / map 相位**：`_store_observation` 保存 `env.step/reset` 返回的 `history`；`_frame` 的 `refreshed = map_cache.steps[0]==0` 与 `native_sample_id` 相位一致；`S2PhysicsEnv` 的 ablation/real evidence 用同一 `refreshed`/`sample_id`。未发现 FIFO 二次写入或地图二次刷新。
- **first_step**：`S2PhysicsEnv` 已改为 `first_step = frame.route_x >= 0.0`（不再是 `top_arrived`），S2 stairs 的 APPROACH→CLIMB 顺序恢复；fake/stairs 复测 enter_climb 不再缺失。

## 2. smoke-r9 证据复算

文件：`logs/a2_v09/s2_0/evidence/smoke-r9/trace.jsonl`，SHA256 `3c06e549759773afa9a5c64e41a38a3f625e0aafd28c107a5d97f8b267a05794`（64 行、27 字段）。

| 量 | 独立复算 | summary | 判定 |
|---|---:|---:|---|
| route_x start/end/delta | -1.0849339962 / -0.7065734267 / **0.3783605694770813** | 同 | 通过 |
| route_x 增量求和 | 0.3783605694770813 | — | 通过 |
| velocity_route 均值 | **0.2983198637375608** | 同 | 通过 |
| a_request 非零帧 | **64/64** | 64 | 通过 |
| foot_contact 每帧和均值 | **2.5625**（脚均值 .6719/.6250/.6094/.6563） | 2.5625 | 通过 |
| g_level 唯一值 | **1.0**（flat） | — | 通过 |
| terminal / failure 帧 | **0 / 0** | — | 通过 |
| pitch 范围（trace 为 rad） | -0.043981… / 0.003929…（-2.52° / 0.23°） | -2.5199° / 0.2251° | 通过 |
| trace 与 `00_flat_walk_real_trace.jsonl` | 字节相同 | — | 通过 |

- `00_flat_walk_real.json` 的 `case_sha256=aee59c5d…` 是 **S2_DIAG flat_walk 0.3 case0**；`case_id=0`，`task=flat_walk`。
- **证据绑定缺口**：`smoke_summary.json` 只有 `trace_path`，`00_flat_walk_real.json` 只有 `npz_sha256`；两者都没有 `trace_sha256`。SHA 仅在外部核对时成立，证据文件本身未绑定（B-V8-4）。

## 3. S1 参考对照（边界）

- S1 参考 trace：`logs/a2_v09/s2_0/parity/s1_flat_case0/trace.jsonl`，首帧 `base_xy=(-1.001377,-0.025564)`，21 字段；S2 smoke-r9 首帧 `base_xy=(-1.084934,-0.038502)`，27 字段。
- 二者**不是同一病例**：S1_DIAG flat case0 与 S2_DIAG flat case0 的 suite/streams 不同，spawn 相差约 0.0836 m；几何同为 flat、速度同为 0.3 m/s，但初态不同。
- 因此"S1/S2 观测差 ≤4.77e-7"只能理解为**同一观测输入重放**下的网络数值对照；不能作为同病例轨迹/控制等价证据。S1 参考 trace 也不能替代 S2_DIAG case 的配对对照。
- 若需同病例对照，应固定同一 `case_sha`/spawn 重放，或在 parity 报告中明确"仅同观测，不同病例"。

## 4. 台账

- `logs/a2_v09/s2/ledger.jsonl`：header run_id `a2-v09-s2-q021-r1`，stage_reserve plan SHA `773b53c6…`；`state()`：envelope **263,888,370**，spent `{engineering:384}`，reserved `{0}`，plan SHA 一致。
- `s2_0_matrix_two_branches`：limit 420,390、spent **384**、reserved 0。
- 事件：10 reserve / 7 commit / 3 release / 1 refund；operation_id 全部唯一；r1 commit 64 后 refund 64（净 0）；r2/r5/r6 reserve→release（空尝试正确释放）；r3/r4/r7/s1ref/r8/r9 各 commit 64（6×64=384）。
- **阻断 B-V8-2**：S1 parity 64 与 5 次 smoke（r3/r4/r7/r8/r9，共 320）全部记在 `s2_0_matrix_two_branches` 上。该 obligation 的上限是 420,390（完整 150 分支上界），再跑满矩阵会变成 420,390+384=420,774 > 420,390；S2.0 子额度 429,822 = 矩阵 420,390 + H 9,432，也会被超 384。应把 smoke/parity 作为独立 engineering obligation（或纳入 S2.0 子额度余量），不能侵蚀矩阵上界。
- 其余：无重复计费；r1 退款、r2/r5/r6 释放均正确结算；envelope 未被改写。

## 5. case 选择语义

- `S2_DIAG` 有 **15 个 `case_id=0`**：flat 0.3/0.5/0.6、stairs 0.3/0.5/0.6（两宽）、platform 0.3/0.5/0.6（两宽）。
- `scripts/run_a2_v09_s2_0.py` 的 `--case`：preflight 用 `next(row for row in matrix['cases'] if row['case_id']==args.case)`，smoke 用 `selected[0]`；两者都取清单中第一个匹配 → **flat_walk 0.3**。`--smoke --case 0` 实际跑的是 flat（证据文件名 `00_flat_walk_real`），不是 stairs_walk。
- 未发现 `--task`/`--case-sha`/group 选择参数或文档说明；与自述"`case_id=0`（stairs_walk）"不符。**B-V8-1（阻断）**：`case_id` 是组内编号而非全局唯一键，smoke/诊断选择必须按 `case_sha` 或 `(task, group, case_id)`；正式矩阵按 batch 的 `case_sha256s` 执行不受影响，但 smoke/parity 的病例归属错误。

## 6. 判定

- **通过**：代码动作/观测根因修复；smoke-r9 数值与 SHA；台账无重复/空尝试结算；history/map 相位。
- **阻断**：
  - B-V8-1：`--case 0` 非唯一，smoke 跑 flat 而非 stairs；正式矩阵未受影响但 smoke/parity 归属错。
  - B-V8-2：S1 parity + smoke 384 记入 `s2_0_matrix`，满矩阵会超 obligation 420,390 与 S2.0 子额度 429,822。
  - B-V8-3：S1 参考与 S2 smoke 非同病例（spawn/suite 不同），"观测等价"边界未在证据中说明。
  - B-V8-4：trace SHA `3c06e549…` 未绑定进 summary/meta（仅 npz SHA）。
- 待裁定：`--case` 的正式选择键；S1 parity 的独立 obligation；smoke 是否应使用 stairs case0（或其他 case_sha）。

机器可读明细见 `logs/a2_v09/s2_0/verification/execution_audit_20261122_r8.json`。
