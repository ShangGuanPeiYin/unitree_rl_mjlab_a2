# V8.5 聚焦复核：platform spawn_z 修复、平台单例、r2 resume 与平台失稳判断

- 复核角色：A2-S2-VERIFY（独立于实现方）
- 复核提交：`7712535 fix(s2): platform spawn uses platform-height z instead of fixed 0.4`
- 复核范围：`spawn_z` 代码/测试、`q032-platform-spawn-r1` 单例、`matrix-full-r2` resume、`s2-040` 平台例、平台 60–100 步失败的归因、matrix 实时预算
- 方式：**纯 CPU 只读现有代码/JSON/trace**；未运行 GPU、未推进物理、未训练、未修改实现/测试/SDS/ledger/checkpoint，未做 git 写入
- 快照：2026-11-22 17:00 CST；matrix r2 仍在运行
- 产物：本报告 + `logs/a2_v09/s2_0/verification/execution_audit_20261122_r85.json`

## 0. 结论摘要

| 项目 | 结论 |
|---|---|
| `MjlabS2Backend` spawn_z | **通过（代码）**：每次 reset 前设置 `default_root_state[:,2]=spawn_z` |
| `S2TrainEnv` 平台 spawn | **通过（代码/计划）**：platform 角色 `platform_height+0.4`；flat/stairs/l40 仍 0.4 |
| 每次 reset 的断言 | **不足**：只在 plan 测试断言 platform `spawn_z`；无 backend/S2TrainEnv reset 测试，无 flat/stairs 回归断言 |
| 平台单例证据 | **通过（有 traceability 缺口）**：spawn z=8.4 来自 case source；trace 只有 feet_z/contact，未持久化 root z |
| 失败分类 | **正确**：59 步 `posture`，非 `policy_not_applied`；terminal/failure/post_reset 一致 |
| r2 resume | **通过**：r1 49 completed + 20 failed + 2 prior skipped；r2 跳过已完成、只重跑 platform；无重复计费 |
| `s2-040` | **通过**：92 步、91 帧非零动作、接触 1.674、on-platform、nonfoot failure |
| 平台快速失败判断 | **不是 spawn/装配/接触 bug**；是 real policy/real-map 行为下的真实失稳；但 plane map 不同，不能仅凭此证明“只有权重问题” |
| matrix 预算 | **未接近上限**：net 62,700 / limit 420,390（约 14.9%）；无超限、无重复计费 |

## 1. spawn_z 修复代码核查

### 1.1 `MjlabS2Backend`

`src/tasks/velocity/v09/s2_physics.py`：

```python
self.spawn_z = float(spawn['local_position'][2]) if len(spawn.get('local_position', ())) > 2 else 0.4
...
def reset(self):
    if hasattr(self, 'spawn_z'):
        self.env.robot.data.default_root_state[:, 2] = self.spawn_z
    obs, _ = self.env.reset()
```

- 每次 `reset()` 都先写 `default_root_state[:,2]`，再调用 `V07Env.reset()`；`V07Env._reset_ids()` 里 `state = robot.data.default_root_state[ids].clone()`，随后 `local[:,2] = state[:,2]`，因此 root z 会取到 `spawn_z`。
- flat/stairs 的 case `spawn.local_position[2]` 是 0.4，仍为原值；缺失 z 时 fallback 0.4，不会把非平台任务抬高。
- 未设置/无 `local_position` 的路径用 0.4；与 Q030/Q032 前的 flat/stairs 行为一致。

### 1.2 `S2TrainEnv`

`src/tasks/velocity/v09/s2_train_real.py`：

```python
if role['role'] == 'platform_walk':
    spawn_z = float(geometry['platform_height']) + 0.4
else:
    spawn_z = 0.4
...
self.env.robot.data.default_root_state[:, 2] = torch.tensor(
    [row['spawn_z'] for row in self.plan['rows']], ...)
self.env.reset()
```

- platform 角色按 `platform_height+0.4`；其他角色 0.4。
- 独立 CPU 扫描 `build_env_plan(1024)`：platform 102 行中 2.0 m 高→2.4（51 行）、4.0 m 高→4.4（51 行）；flat_walk/flat_stop/stairs_walk/l40 全部 0.4；坏行 0。
- **但** `S2TrainEnv.reset()` 没有再次写 `default_root_state`，只在构造后写一次。功能上 V07 每次 reset 从该 buffer clone，故通常仍生效；若其他路径改写该 buffer，则没有 per-reset 再断言。此点列为 R-V85-2。

### 1.3 断言/测试覆盖

- 新增测试只有 `test_real_env_plan_maps_roles_to_s2_terrain` 中一行：
  `assert row['spawn_z'] == row['geometry']['platform_height'] + 0.4`。
- 独立运行该测试：**1 passed**（CPU）。
- **没有**以下测试：
  1. `MjlabS2Backend.reset()` 连续两次 reset 后 `default_root_state[:,2]` 仍为 `spawn_z`；
  2. `S2TrainEnv` 构造后 `default_root_state` 的 102 个平台值正确；
  3. flat/stairs/l40 的 `spawn_z` 仍为 0.4 的显式回归断言（当前只能靠 plan scan 发现）。
- 因此“231 tests pass”不等于 spawn 修复有 reset 级覆盖；列为 **B-V85-1（测试覆盖）**。代码本身未发现功能性错误。

## 2. 平台单例 `q032-platform-spawn-r1`

case SHA：`b128038d454c9f150ddeeed6c5ecf1fcf5e5f8035953e5b075205f9dfd92fd6d`。

case source（`S2_DIAG.json`）确认：

```text
spawn.local_position = [9.0, -0.03761838836799239, 8.4]
spawn_surface = top_platform
height_contract = top_surface_z + 0.4
platform_height = 8.0
```

证据：

| 项 | 值 |
|---|---|
| control_steps / trace rows | 59 / 59（step 0..58） |
| 首帧 feet_z | 8.0483, 8.0445, 8.0235, 8.0257 |
| 首帧 foot_contact | [F, F, T, T]；均值 1.8983 |
| 首帧 a_request | 12 维非零 |
| action_nonzero_frames | 58 / 59 |
| `policy_calls` | 59 |
| termination_reason | `posture` |
| failure | true；event step 59 |
| terminal row | step 58，terminal=true，post_reset=true |
| `terminal_row_is_post_reset` | true |
| trace SHA | `f259d897d205114964ea1f2ee661bd57932baac8b338a1ffa853f62a22de3cc9` |
| npz SHA | `45031df935a0b06bcd189b1b153d5509ccc9e0d3e08960ab832fbcbb926e5fac` |

- `status=completed`、`error` 不存在；`action_nonzero_frames=58`，`policy_calls=59`，因此**不是** `policy_not_applied`。
- failure event 与 trace 的 `failure=true`、`terminal=true`、`post_reset=true` 一致；`pre_terminal_sample` 指向 row57。
- **注意**：“四足接触 1.9”是 59 帧的平均接触足数 1.898，不是“四足同时接触”；首帧只有 2/4 足接触，随后逐步增加。如果验收要求首帧 4 足接触，本条不满足；如果只要求“有接触、非零动作、平台起步”，则通过。
- **traceability 缺口**：summary/trace 没有 `spawn_z`/root z 字段；8.4 只能由 case source（case_sha 绑定）确认，trace 只能看到 feet_z≈8.02–8.05。列为 R-V85-1，不影响本次数值判断，但后续报告不应把“spawn z=8.4”写成 trace 直接实测。

## 3. `matrix-full-r2` resume 行为

### 3.1 状态计数

r1（`progress-matrix-full-r1.jsonl`）：

```text
completed=49, failed=20, skipped_completed=2
failed 全部是 platform（s2-040..049, 060..069），各 1 步
```

r2 快照（17:00 CST）：

```text
skipped_completed=50, completed=10 (platform reruns), failed=0
last processed: s2-059-plane_diagnostic-d900c3767344 (skipped)
```

- r1 的 49 completed + 2 prior `skipped_completed` 都会被 r2 跳过；因此若把所有 `skipped_completed` 都算作“跳过”，最终会是 **51**，而“49”是 r1 completed 的数目。用户所述“49 条跳过”应理解为 49 条 completed，另有 2 条 prior skipped；不是重复/重跑。
- r2 当前跳过 50 条已处理项，随后继续跳过 s2-060..069 之外的已完成 stairs（含 s2-071）；平台 20 条全部重跑。
- 未发现任何 r1 completed 的条目在 r2 重新执行；`valid_completed` 的 status/case_sha/evidence SHA 校验路径工作正常。

### 3.2 台账与失败 release

- r1 platform failed 每条：`reserve 2167 / commit 1 / release 2166`，对齐。
- q032 单例：`reserve 1500 / commit 59 / release 1441`，对齐。
- r2 每条 platform：`reserve 2167 / commit actual / release remainder`，对齐；无 failed 未 release。
- 无重复计费：r2 对 skipped 条目没有 reserve/commit；只对 platform 重跑写入。

## 4. `s2-040` 平台例抽查

路径：`logs/a2_v09/s2_0/matrix/s2-040-real-f6fa8fbdf3fc/r2`；case SHA `f6fa8…`。

| 项 | 值 |
|---|---|
| case spawn | [9.0, 0.0076255957071354324, 8.4]（case source） |
| control_steps | 92 |
| termination_reason | `nonfoot` |
| failure | true |
| action_nonzero_frames | 91 / 92 |
| first action | 12 维非零 |
| contact_feet_mean | 1.6739 |
| 首帧 feet_z | 8.0480, 8.0450, 8.0234, 8.0277 |
| 首帧 foot_contact | [F, F, T, T] |
| settled | committed 92 / released 2075 / actual 92 |

- 与 q032 单例一致：平台起步、动作非零、有接触，但 real 分支在 92 步以 nonfoot 结束；不是 spawn 失败或 `policy_not_applied`。
- 平台 trace 同样没有 root z / spawn_z 字段。

## 5. 平台 60–100 步 failure 的独立判断

### 5.1 事实

real 平台例（r2 当前）：

| 例 | steps | failure | 动作非零帧 | contact mean |
|---|---:|---|---:|---:|
| q032 单例 | 59 | posture | 58 | 1.898 |
| s2-040 | 92 | nonfoot | 91 | 1.674 |
| s2-042 | 85 | posture | 84 | 1.918 |
| s2-044 | 95 | nonfoot | 94 | 2.421 |
| s2-046 | 59 | posture | 58 | 1.847 |
| s2-048 | 39 | nonfoot | 38 | 2.077 |

plane 同 case：

| 例 | steps | 结果 |
|---|---:|---|
| s2-041 | 785 | success，净前进 5.003 m |
| s2-043 | 790 | success，净前进 5.001 m |
| s2-045 | 786 | success，净前进 5.002 m |
| s2-047 | 788 | success，净前进 5.003 m |

### 5.2 判断

- **不是装配/出生/接触 bug**：root z 由 case source 8.4 给出；首帧 feet_z≈8.02–8.05，四足都在平台上；动作非零；足接触非空；失败分类为 `posture`/`nonfoot`，不是 `policy_not_applied`。
- **是 real 分支的真实失稳**：同样的 platform 装配、同样 spawn 起点，plane 分支能走完 5 m；real 分支在 39–95 步内 posture/nonfoot。说明物理 reset/接触链已能运行，失败发生在策略/真实图输入层面。
- **不能只归因于权重**：real 与 plane 的差异包含地图输入（Q031 的 plane 替换）与分支行为；当前没有同帧双输入证据，不能写“仅权重问题”。独立表述应为：**real policy + real-map input 的真实失稳，不是 spawn/装配/接触问题；权重-only 未证明**。
- 这也解释了为什么修复后不再是 1 步 fall；修复解决的是出生高度，不是平台起步的策略稳定性。

## 6. matrix 预算（17:00 快照）

| 项 | 值 |
|---|---:|
| matrix net | **62,700**（gross 63,148，refund 448） |
| full 已完成步数 | 59,503（r1 55,196 + r2 4,307） |
| limit2 steps | 3,177 |
| 残差 | 20（历史小额，未发现重复） |
| debug net | **10,235** |
| 其中 pair+dim | 1,974 |
| 其中 parity dim4/64 | 136 |
| 其中 q032 单例 | 59 |
| H | 0 |
| reserved outstanding | 0（当前无活跃预留） |
| matrix limit | 420,390 |
| 使用率 | 约 14.9% |
| 是否接近上限 | **否** |
| 是否超限/重复 | 否 |

- r1 platform 20 条失败各只 commit 1 步，r2 重跑实际步数另计；总账无重复。
- q032 单例 debug 59 步；dim4/dim64 136 步已在 ledger 单独 commit（上轮缺口已补）。
- matrix 距离上限仍有约 357k 步余量；整批继续，无需因预算暂停。

## 7. 发现清单

| ID | 级别 | 内容 |
|---|---|---|
| P-V85-1 | 通过 | backend 每次 reset 设 root z；plan 映射 platform=height+0.4、其他=0.4，扫描无坏行 |
| P-V85-2 | 通过（有缺口） | 平台单例 59 步 posture、非零动作、接触 1.898、failure/post_reset 一致，非 policy_not_applied |
| P-V85-3 | 通过 | r2 resume 跳过 completed/prior skipped，只重跑 platform；r1 failed reserve/release 对齐 |
| P-V85-4 | 通过 | s2-040 92 步、91 帧非零、contact 1.674、nonfoot failure |
| P-V85-5 | 判断 | 平台快速失败是 real policy/real-map 失稳，不是 spawn/装配/接触；权重-only 未证明 |
| P-V85-6 | 通过 | matrix 62,700/420,390，无超限/重复；debug 10,235 |
| B-V85-1 | 阻断（测试覆盖） | 仅 plan 测试；无 backend 每次 reset、S2TrainEnv default state、flat/stairs 0.4 回归测试 |
| R-V85-1 | 待澄清 | 单例 summary/trace 未持久化 spawn_z/root z；8.4 由 case source 确认 |
| R-V85-2 | 待澄清 | S2TrainEnv 只在构造后设一次 default_root_state，reset 不再断言 |
| R-V85-3 | 待澄清 | 49 是 r1 completed；另有 2 prior skipped_completed，r2 全量统计可能显示 51 skipped |

## 8. 证据清单

- `src/tasks/velocity/v09/s2_physics.py`（`spawn_z`/`reset`）
- `src/tasks/velocity/v09/s2_train_real.py`（plan `spawn_z`、`S2TrainEnv` 设置）
- `tests/test_v09_s2_1_train.py::test_real_env_plan_maps_roles_to_s2_terrain`
- `logs/a2_v09/s2_0/evidence/q032-platform-spawn-r1/{smoke_summary.json,trace.jsonl}`
- `logs/a2_v09/s2_0/cases/S2_DIAG.json`（case `b128…`/`f6fa…` spawn z=8.4）
- `logs/a2_v09/s2_0/matrix/s2-040-real-f6fa8fbdf3fc/r2/{smoke_summary.json,trace.jsonl}`
- `logs/a2_v09/s2_0/matrix/progress-matrix-full-r1.jsonl`
- `logs/a2_v09/s2_0/matrix/progress-matrix-full-r2.jsonl`
- `logs/a2_v09/s2/ledger.jsonl`
- `logs/a2_v09/s2_0/verification/execution_audit_20261122_r85.json`

## 9. 收口

- spawn_z 修复的代码路径正确，平台单例与 s2-040 均证明出生高度已生效；失败不再是 `policy_not_applied`。
- 平台 60–100 步失败不是出生/接触装配问题；证据支持 real policy/real-map 的真实失稳，但不足以把原因单独归给权重。
- r2 resume 逻辑正确；49 completed 与 2 prior skipped 的计数口径需在报告中写明，不能把 51 误报成重复。
- 测试覆盖不足是当前主要证据缺口；建议补 backend 连续 reset、S2TrainEnv default state、flat/stairs 0.4 回归三个测试。
- matrix 预算安全（约 14.9%），继续运行；接近上限或出现失败未 release 时立即报告。
