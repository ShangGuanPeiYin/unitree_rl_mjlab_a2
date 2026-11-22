# A2 v0.9 S2 执行链路 V8.1 独立复核（台账扩展/T10/方法学）

- 审核 agent：A2-S2-VERIFY（独立验收子 agent，未参与实现）
- 审核时间：2026-11-22（快照 15:20–15:30）
- 仓库/分支/HEAD：`/home/xxs/dog/unitree_rl_mjlab_a2` / `main` / `5ef8b6e`
- 范围：台账 `stage_extend`、case 选择、证据 SHA、T10 逐帧结果、real/plane 方法学、S1 parity；纯 CPU/只读证据。
- 机器可读件：`logs/a2_v09/s2_0/verification/execution_audit_20261122_r81.json`

## 0. 结论

| 项 | 结论 | 要点 |
|---|---|---|
| 1 台账 stage_extend | **通过（附残留风险）** | envelope 263,908,370 = 11 项限额和；matrix 420,390/384 未改；debug 20,000/7,682；spent 8,066、reserved 0；重复/超限拒绝 |
| 2 case 选择 | **通过** | `--case-sha` 严格前缀；`--case-id` 必须 task/speed/platform-width；歧义报错；文件名/摘要绑定身份 |
| 3 证据 SHA 绑定 | **通过** | trace/npz/case/attempt/evidence_files 全写入；t10 real-r4、plane-r1 实测一致 |
| 4 T10 真实结果 | **通过** | real enter 174/四足1560/top1563/g>0 1564/g=1 1588/body failure 1608、pitch −53.19、route_max 9.27；plane s−t=1/首替换1529/valid 1.0/saturation .025/nonfoot、pitch 均值13.36 |
| 5 real/plane 方法学 | **阻断** | 独立进程 action replay 仍有 route 4.93e-4、qvel .728 漂移，prefix 不等；不能支持 Q016 同状态配对；T11 未交付 |
| 6 S1 parity | **阻断/待裁定** | 同 spawn/流/命令的 S1 代理；strict 1e-6 失败（step3 proprio 2.62e-6，max 1.75e-4）；r4 practical=false(1e-5)、r6 practical=true(1e-3) 仅靠放宽容差 |
| **总判定** | **台账/病例/证据/T10 通过；Q016 配对与 parity 阻断** | 可继续非物理证据，但不得据 real/plane 独立运行声称 Q016 同状态结论 |

## 1. 台账 stage_extend

- 事件 `stage_extend`（seq 23）：`s2_0_pipeline_debug`，20,000，category engineering，`subcap=S2_0`，basis “SDS §4.3 / DSH-V8-2”。
- 当前 state：envelope **263,908,370** = 11 项 obligation 限额之和（263,888,370+20,000）；`spent={engineering:8,066}`、`reserved=0`；`s2_0_matrix` 仍 **420,390/384**，debug **20,000/7,682**。
- 8,066 = matrix 384 + debug 7,682；debug 7,682 = commits 128+67+128+1577+1605+1608+1569+1000，与运行记录一致。
- 历史事件未改写：ledger 链完整，`plan_sha256` 仍为原 `773b53c6…`；扩展只追加。重复 obligation id、超 4 亿扩展实测均 `ValueError`（`extension obligation_id already exists` / `stage_extend exceeds nonlearning cap`）。
- 子额度：debug 计入 S2.0 子额度，429,822+20,000=**449,822** ≤50,000,000；非学习 263,908,370 ≤4 亿；stage 仍 ≤40 亿。
- **残留风险（待裁定）**：原 matrix 上 384（历史 smoke/parity）仍留在 matrix，未迁移到 debug；若跑满 150 分支 420,390，matrix 合计 420,774 > 420,390。stage_extend 只覆盖新 debug，未消除该历史重叠。

## 2. case 选择

- `select_case`：`--case-sha` 支持完整/唯一前缀，多个匹配报歧义；`--case-id` 未同时给 task/speed/platform_width 即报错；`--case` 为弃用别名。
- t10 摘要：suite `S2_DIAG`、case_sha `0140473c…`（stairs 0.8/0.3/case0）、task stairs_walk、speed .3、hard_timeout 6688、requested 2000、实际 1608 终止、geometry/platform_width/front/spawn；证据文件名 `S2_DIAG_00_stairs_walk_0140473c4282_real`。满足身份绑定。
- 文件名未包含 speed/width/timeout，但摘要/JSON 含全部字段；可接受，建议文件名补速度/宽以便肉眼区分。

## 3. 证据 SHA 绑定

- `t10-stairs-real-r4`：trace `4e43188e…`（文件实测一致）、npz `c004effe…`（一致）、case_sha `0140473c…`、attempt_id、4 个 evidence_files；逐例 JSON 同样含 trace_sha/npz/case/attempt/suite/task。
- `t10-stairs-plane-r1`：trace `e34c0c2e…`、npz `94d135d8…`、case 同、attempt/evidence_files 完整；`t10_branch_parity.json` 绑定 real/plane 两条 trace SHA。
- 结论：**通过**；`trace_json_sha256` 字段存在。

## 4. T10 逐帧复算

### real（1608 行，trace SHA `4e43188e…`）

| 量 | 独立复算 |
|---|---:|
| enter_climb | **174**（route 0.0017） |
| 四足入台 | **1560**（route 8.3368） |
| top_arrived | **1563**（route 8.4327） |
| g>0 / g=1 | **1564 / 1588** |
| body failure 事件 | **1608** |
| route_max | **9.265788** |
| 失败前 pitch | **−53.1867°**（行 1607，step 1606，route 9.2658） |
| 最后一行 | 行 **1608**，step 1607，route −1.0486，terminal=True，**auto-reset 后出生点** |

**真实失败采样行号：1607（1-based；trace step 1606）；最后一行 1608 只是 reset 后出生点。** 事件步 1608 与物理采样行 1607 差一帧，引用时必须区分。

### plane（1569 行，trace SHA `e34c0c2e…`）

| 量 | 独立复算 |
|---|---:|
| enter_climb | 174 |
| four/top | 从未（None） |
| s−t / first replacement | **1 / 1529** |
| valid_fraction | **1.0**（1,882,800/1,882,800） |
| saturation | **0.025**（1,200/48,000） |
| route_max | **8.146980** |
| nonfoot failure 事件 | **1569** |
| 失败采样行 | **1568**（step 1567，route 8.1440，pitch **+15.91°**） |
| 最后一行 | 1569，auto-reset 出生点 |
| platform 后 pitch | 均值 **+13.36°**、最大 +15.91°（“+13.4°”是均值口径） |

- real/plane 独立进程前缀：action replay 相同动作下 route 最大差 **4.9257e-4**、joint_velocity **0.7277**、pitch 4.31e-3、`exact_trace_prefix_equal=false`；说明独立进程/双 env 求解顺序导致状态漂移，**不是同物理状态**。

## 5. 方法学与 T11

- Q016 要求“同一物理状态、同一采样帧”的真实/替换对照。当前 real/plane 是**独立进程**分别推进，即使用同一 action 重放，route/qvel 已漂移（4.93e-4 / 0.728），不能作为同状态配对证据。
- 同进程配对（T11）在原理上可满足：一次物理推进、同一状态同时生成真实图与替换图、同一 `native_sample_id`/采样帧；但当前 `S2PhysicsEnv` 是单分支 env，T11 尚未交付，现有 replay300 仍是第二进程。**结论：T11 之前，Q016 配对证据阻断。**
- 不能以“动作相同”替代“状态相同”；`exact_trace_prefix_equal=false` 已明确暴露。

## 6. S1 parity

- S1 代理：`s1_proxy_case_sha e3887…`（suite `PARITY_S1_PROXY`），但 spawn/streams/command 与 S2 case `aee59c5d…` 相同（spawn −1.085850、yaw 0.012177、同流/命令）；输入层同病例，case_sha/suite 身份不同。
- r6：观测最大差 proprio **1.746e-4**、history 同、heightmap **5.96e-8**、route **1.624e-6**；64 帧仅 2 帧哈希相同；首次 strict 分歧 step 3 proprio **2.62e-6**（容差 1e-6）→ `strict_parity_ok=false`。
- **practical 口径变化**：r4 容差 1e-5、max proprio 2.66e-4 → `parity_ok=false`；r6 容差改为 1e-3、max 1.75e-4 → `parity_ok=true`。同一量级差异通过放宽 100 倍容差变成“practical pass”，没有裁定依据；**这不是可接受的等价证明**。
- 1.75e-4 的 proprio 差远大于 float32 噪声（~1e-7），首次分歧 step 3 已 2.6e-6；更像相位/观测管线差异（S1 48×32 vs S2 40×30、history/map 刷新顺序），**不能归因于 GPU 浮点噪声**。strict 1e-6 失败是真实分歧；practical=true 只反映 1e-3 容差。

## 7. 判定

- **通过**：stage_extend 账务/caps/矩阵不变/重复与超限拒绝；case 选择；证据 SHA 绑定；T10 数值与失败行；Q027/g 事件。
- **阻断**：
  - B-V81-1：real/plane 独立进程漂移（route 4.93e-4、qvel .728）不能支持 Q016 同状态配对；T11 未交付。
  - B-V81-2：parity practical 容差由 1e-5 放宽到 1e-3 才 pass，strict 1e-6 失败；1.75e-4 不是浮点噪声，practical=true 不可作为等价证明。
  - B-V81-3：matrix 历史 384 未迁移，满矩阵 +384 超 420,390；stage_extend 未消除该重叠。
- **待裁定**：T11 同进程配对；parity 容差权威；matrix 历史 384 的归类/追加。

机器可读明细见 `logs/a2_v09/s2_0/verification/execution_audit_20261122_r81.json`。
