# S2 T10：病例选择、调试台账、S1 同病例 parity、楼梯 real/plane

日期：2026-11-22  
范围：DSH-V8（B-V8-1..4）与 T10。未训练、未改 S1 文件、未改 SDS、未提交 git。

## 1. 病例选择语义（B-V8-1）

`S2_DIAG.json` 有 75 个 native case；`case_id` 只在组内为 0..4，跨 flat/stairs/platform 和速度/宽度重复。`case_id=0` 共有 15 个，所以选择禁止按列表下标或“第一个 case_id 匹配”。

`scripts/run_a2_v09_s2_0.py` 现在：

- `--suite S2_DIAG`（默认）或 `--suite H_SERIES`；
- `--case-sha <sha256>` / `--case-sha256`：按完整或唯一前缀严格匹配 `case_sha256`；
- `--case-id N` 必须同时提供 `--task`、`--speed`、`--platform-width`，否则报“not globally unique”并拒绝；
- `--case` 仅保留为 `--case-id` 的弃用别名，语义相同，不再取列表首项；
- 多匹配时报出所有候选，不静默取第一个；
- `--steps` 会被钳制到该病例的 `hard_timeout_steps`，summary 同时记录 requested/effective。

T10 目标病例：

```text
suite=S2_DIAG
case_id=0
case_sha256=0140473c4282e721cceec4f0fcd05536337869b9b0379c9947a4059224012150
task=stairs_walk
stair_count=40
h=0.2, step_depth=0.2, platform_width=0.8
speed=0.3
hard_timeout_steps=6688
platform_front_local_x=8.0
spawn=(-1.0486490, -0.0254329)
```

等价的严格选择命令：

```bash
python scripts/run_a2_v09_s2_0.py \
  --execute --smoke --suite S2_DIAG \
  --case-id 0 --task stairs_walk --speed 0.3 --platform-width 0.8 \
  --steps 2000 --device cuda:0 --branch real \
  --attempt-id t10-stairs-real-r4 \
  --obligation-id s2_0_pipeline_debug \
  --ledger logs/a2_v09/s2/ledger.jsonl
```

summary 顶层现在包含：`suite / manifest_suite / source / case_id / case_sha256 / task / stair_count / speed / hard_timeout_steps / geometry / platform_front_local_x / spawn_x / spawn_y`。

逐例证据文件名改为：

```text
{S2_DIAG|S1_DIAG}_{case_id:02d}_{task}_{case_sha256前12位}_{branch}.npz
..._trace.jsonl
...json
```

逐例 JSON 与 `smoke_summary.json` 均记录 `attempt_id / case_sha256 / npz_sha256 / trace_sha256`；summary 额外列 `evidence_files` 及每个文件的 SHA256。

## 2. 调试台账扩展（B-V8-2）

`src/tasks/velocity/v09/s2_ledger.py` 新增 append-only `stage_extend`：

- 只追加事件，不修改历史行或原 10 项义务的 limit；
- 新义务必须 `category=engineering`，`subcap` 可为 `S2_0`；
- 抬升 envelope 恰好等于新义务额度；
- 校验 nonlearning ≤ 4e8、总 stage ≤ 4e9、S2.0 子额度 ≤ 5e7；
- 重复 obligation_id/name 拒绝；
- CLI：`scripts/extend_a2_v09_s2.py --samples ... --write`。

已在正式台账执行一次：

```text
stage_extend seq=23
obligation_id=s2_0_pipeline_debug
samples=20000
envelope: 263888370 -> 263908370
matrix limit: 420390（不变）
matrix spent: 384（历史保留，未迁移）
event_sha256=36810b1abfe27f863336a21c3a31a2e05c431efc446e8e67a4752da1e0ca8970
```

`run_a2_v09_s2_0.py` 的 smoke/parity/重试默认 `--obligation-id s2_0_pipeline_debug`；正式 150 条矩阵必须显式传 `--obligation-id s2_0_matrix_two_branches`。H 六例使用既有 `h_series_baseline_supplement`（limit 9432）。

T10 结束时：

```text
s2_0_matrix_two_branches: limit 420390, spent 3177（limit-2 real+plane）, available 417213
s2_0_pipeline_debug:      limit 20000, spent 8066, available 11934
h_series_baseline_supplement: limit 9432, spent 0, available 9432
```

## 3. 同病例 S1/S2 parity（B-V8-3）

旧 r9 parity 的 S1 是 `S1_DIAG` flat case0，S2 是 `S2_DIAG` flat case0；spawn/streams 不同，不能作同病例轨迹对照。

新增 `scripts/run_a2_v09_s2_parity.py`：从 S2 病例复制 spawn/yaw/streams/command，构造 `V09S1Env` 的 **flat_walk flat tile 0** 代理；S1 与 S2 接收同一 S1 Actor 动作序列，逐帧记录观测 SHA256 与逐键最大差。

flat case 结果（`aee59c5d3a49...`，4 步精确诊断；64 步旧 r6 仅留痕）：

```text
s2_case_sha256 = aee59c5d3a49a300c85a0a0067841059d36ea283ce1c0e62589477444e8bcef2
s1_proxy_case_sha256 = e3887ccdb0503b76f9fac2d4b88ba854b0ec34cbae9426d5be5ed6220b0b9d2e
spawn = (-1.0858503909, -0.0385040479)

step0: qpos/qvel hash、proprio、history、heightmap、route_error 全一致
step1: qvel index13 |Δ|=4.768e-7；qpos max |Δ|=7.45e-9
       proprio dims 27/28/30/32（joint_vel 6/7/9/11）1.2e-7..4.8e-7
       history dims 207/208/210/212（最新 history 帧的同一组 joint_vel）
       heightmap/route_error 0；a_request/a_applied 0
step2/3: 差异仍是 joint_vel 及其 history 副本；heightmap/route_error 0
parity_ok(1e-6) = false
```

结论：不是 q-q0、omega_b/gravity_b 坐标系、last_applied_action、normalizer、
history 写入时机或 map 采样先出错；第一次差异是物理步后的 qvel index13，随后被状态放大到 1e-4 量级。来源定位在物理状态，不是可修补的观测管线；引擎级非确定性/接触求解尚未定位到可修改代码。旧 r6 的 1e-3 “practical pass” 作废，不再使用放宽容差。

证据：

```text
logs/a2_v09/s2_0/parity/t10/aee59c5d3a49/t10-parity-flat-r6/
  parity_summary.json
  s1_reference.jsonl
  s2_reference.jsonl
```

适用范围：flat 病例可用于整段同病例对照；楼梯病例的 S1 代理是平地环境，只在第一次楼梯接触前有对照意义，不能当作楼梯轨迹对照。

## 4. T10 真实楼梯 real/plane

共同病例：`0140473c4282...`，2000 步请求，real/plane 各自独立进程。

### 4.1 real（`t10-stairs-real-r4`）

```text
status=completed（2000 步前 v07 body failure）
control_steps=1608
route_x: -1.04767 -> max 9.26579（末帧为 v07 auto-reset 的出生点）
enter_climb: step 174, route 0.00174, t=3.48 s
four_feet_platform: step 1560, route 8.33685, t=31.20 s
top_arrived: step 1563, route 8.43270, t=31.26 s
g_level >0: step 1563; g_level=1: step 1587
failure: step 1608, reason=body
末次未 reset 采样: step 1606, route 9.26579, pitch -53.19°
pitch 后段（Q032：trace step 1562..1606，45 帧，剔除复位行）:
  mean -25.6311°, min -53.1867°, max -11.1802°；旧 44 帧/-25.96° Q030 标 superseded
  nose_down_fraction=1.0, nose_up_fraction=0.0, level_fraction=0.0
四足每足最大高度约 8.12–8.25 m，四足均完成抬升
```

判断：确实上梯并到达平台顶部；Q030 口径下 pitch<0 是低头/前倾，因此 real 在平台后持续低头（前倾），不是抬头/后仰；约 route 9.27 m 处 body failure。

窗口/事件对照（event step 与 trace row 分别列出；Q030 行配对见重算 JSON）：

| case | 机身首次 route>=8.0 | 四足入台 | top_arrived | §6.5 正式巡航窗 |
| --- | --- | --- | --- | --- |
| real-r4 | step 1546 / row 1546 | event 1560 / row 1559 | event 1563 / row 1562 | NA（D_post 5 m 未达到即 failure） |
| plane-r1 | step 1550 / row 1550 | NA | NA | NA（未 top，nonfoot failure） |

### 4.2 plane（`t10-stairs-plane-r1`）

```text
status=completed（failure）
control_steps=1569
route_x max=8.14698
top_arrived=false
failure: step 1569, reason=nonfoot
plane trigger_frame=1528
first_replacement_frame=1529
first_consumed_replacement_frame=1529
s_minus_t=1
valid_fraction=1.0 (1882800/1882800)
saturation=0.025 (1200/48000)
replacement_frames_consumed=40
pitch 前缘后（step 1550..1567，18 帧，剔除复位行；首次 route>=8.0 为 step1550）:
  mean +14.10°, min +12.23°, max +15.91°
  nose_down_fraction=0.0, nose_up_fraction=1.0, level_fraction=0.0
```

plane 在平台前缘附近（约 8.15 m）失败，未锁存 top；Q030 口径下 +pitch 是抬头，plane 在平台后持续抬头，real 则是持续低头/前倾。旧 +13.36°/94.7% 含复位行且方向误标，作废。

real-r4 的 `post_reset` 旧 trace 无字段，sidecar 标记 row 1607=true；plane-r1 row 1568=true。新代码已把 `post_reset` 写入每帧。

### 4.3 分支前缀

normal real/plane 是独立 GPU 进程；同一动作序列 replay 证明 `a_request`/`a_applied` 逐帧完全一致，但 qpos/qvel 哈希在 step 2 就不同（同进程顺序重建 env 也一样），说明**病例匹配但物理状态未对齐**，不是 Q016 的同态输入证据。`branch_prefix_sha256` 必须由同一个物理快照/同一采样帧产生；当前跨进程/顺序重建只能声明“动作/命令前缀一致”，不能声明 Q016 满足。详见：

```text
logs/a2_v09/s2_0/evidence/t10-stairs-plane-r1/t10_branch_parity.json
```

该限制应在正式矩阵报告中作为已知限制保留，不能用 4.77e-7 之类的“重放一致”替代同病例/同动作证据。

### 4.4 证据哈希绑定（B-V8-4）

real（`t10-stairs-real-r4`）：

```text
trace.jsonl       4e43188e53e919683a2dc120b8bcc605ed79710ffb124fd9f7d5a0b25105ab4b
npz               c004effe77227e138add82860afaff4e0e5ff2000af0f7257c73c4a3befa9673
case metadata     80210b6caf19d7e77fc2ea192e6ad6a73630844c631676cb6e1bf47927329844
文件名            S2_DIAG_00_stairs_walk_0140473c4282_real.{npz,json}
```

plane（`t10-stairs-plane-r1`）：

```text
trace.jsonl       e34c0c2e7f8a4dad70de4757487377286669eaefd2894daa37c5ab7d19841c5a
npz               94d135d881fe62fe3423d1ce48f86078e890e59eb8a7bfa2471aaae88e906220
文件名            S2_DIAG_00_stairs_walk_0140473c4282_plane_diagnostic.{npz,json}
```

逐例 JSON 与 summary 都含 `attempt_id/case_sha256/npz_sha256/trace_sha256`；summary 的 `evidence_files` 列出 incremental trace、npz、case metadata、case trace 四个文件的 SHA256。

### 4.5 flat 成功路径回归（`t10-flat-1001-r1`）

关闭 v07 success 后验证 flat S2 的 S2 自身完成事件：

```text
case_sha256=aee59c5d3a49...（S2_DIAG flat 0.3）
control_steps=1000
state=SUCCESS, success=true, failure=false, timeout=false
route_x: -1.08493 -> 5.38234, delta 6.46727 m
speed_mean=0.32362 m/s
success event: step 1000, t=20.0 s
settled committed=1000, released=1
```

说明 S2 的 flat 成功仍由 `completion_steps` 在 timeout 前触发，没有被关闭的 v07 success 影响。

## 4.6 Q030 pitch 口径与重算

字段 `pitch` 保留原始值，不重新定义符号。Q030 口径：

```text
pitch = -asin(2(wy-zx)) = asin((R_WB e_x)_z)
positive = nose_up（抬头）
negative = nose_down（低头/前倾）
zero     = level
前向轴 +X，世界向上 +Z
```

证据/报告写入 `pitch_sign_convention`；派生字段改为 `nose_down_fraction`、`nose_up_fraction`、`level_fraction`，零值计入分母。旧 `positive_forward_lean_fraction` 语义错误，旧产物保留不删。

从原始 trace 重算（剔除 auto-reset 出生行，保留终止事件）。固定四窗：

```text
real-r4（Q032 首真帧口径）:
  front_edge: 61 帧, trace_step/row 1546..1606, mean -20.0655°, nose_down=0.9344
  four_feet:  48 帧, trace_step/row 1559..1606, mean -24.6910°, nose_down=1.0,
               latched_event_step=1560
  top_arrived:45 帧, trace_step/row 1562..1606, mean -25.6311°, nose_down=1.0,
               min -53.1867°, max -11.1802°, latched_event_step=1563
  s6_5_cruise: NA（四足后净前进 1 m 未达到，body failure）
plane-r1:
  front_edge: 18 帧, trace_step/row 1550..1567, mean +14.1021°, nose_up=1.0,
              min +12.2248°, max +15.9076°
  four_feet/top_arrived/s6_5_cruise: NA（未四足入台、未 top、未巡航）
```

旧 Q030 的 44 帧/−25.96° 与 18 帧/+14.10° 原件保留但标 superseded；Q032 窗口包含首真帧，不再用事件号/全局 step-1。每窗并列 `trace_step`、`row_index`、`latched_event_step`、`event_completed_steps`。

所有窗口 `post_reset` 行剔除；`step_min/max` 与 `step_row_pairs` 在重算产物中逐条记录。

产物：

```text
logs/a2_v09/s2_0/evidence/t10-stairs-plane-r1/q030_pitch_recompute.json
logs/a2_v09/s2_0/evidence/t10-stairs-real-r4/post_reset.json
logs/a2_v09/s2_0/evidence/t10-stairs-plane-r1/post_reset.json
```

`step/row` 配对逐条列在 `q030_pitch_recompute.json` 的 `step_row_pairs` 中。

## 5. 整矩阵命令（150 + H 6）

正式矩阵必须使用矩阵义务；不要用默认 debug 义务。

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
python - <<'PY' > /tmp/s2_matrix.tsv
import json
p = json.load(open('logs/a2_v09/s2_0/cases/S2_DIAG_run_plan.json'))
for r in p['entries']:
    print(f"{r['map_branch']}\t{r['case_sha256']}\t{r['hard_timeout_steps']}")
PY
python - <<'PY' > /tmp/s2_h.tsv
import json
h = json.load(open('logs/a2_v09/s2_0/cases/h_series_case0.json'))
for r in h['records']:
    print(f"{r['case']['case_sha256']}\t{r['case']['hard_timeout_steps']}")
PY

mkdir -p logs/a2_v09/s2_0/matrix_logs
set -euo pipefail

while IFS=$'\t' read -r branch sha steps; do
  attempt="matrix-$(date -u +%Y%m%dT%H%M%SZ)-${branch}-${sha:0:12}"
  python scripts/run_a2_v09_s2_0.py \
    --execute --smoke --suite S2_DIAG \
    --case-sha "$sha" --branch "$branch" --steps "$steps" \
    --device cuda:0 --attempt-id "$attempt" \
    --obligation-id s2_0_matrix_two_branches \
    --ledger logs/a2_v09/s2/ledger.jsonl \
    >> "logs/a2_v09/s2_0/matrix_logs/${attempt}.log" 2>&1
done < /tmp/s2_matrix.tsv

while IFS=$'\t' read -r sha steps; do
  attempt="h-$(date -u +%Y%m%dT%H%M%SZ)-${sha:0:12}"
  python scripts/run_a2_v09_s2_0.py \
    --execute --smoke --suite H_SERIES \
    --case-sha "$sha" --steps "$steps" \
    --device cuda:0 --attempt-id "$attempt" \
    --obligation-id h_series_baseline_supplement \
    --ledger logs/a2_v09/s2/ledger.jsonl \
    >> "logs/a2_v09/s2_0/matrix_logs/${attempt}.log" 2>&1
done < /tmp/s2_h.tsv
```

预计步数：

```text
S2_DIAG 150 branches: 419,402（逐例 hard_timeout 之和）
H 6 cases:             7,874
total:               427,276
```

预算：limit-2 已用 3,177，matrix available 417,213；尚未跑的 148 条上限约 416,225，余量约 988。H 义务 9,432，可余 1,558。任何重试/额外 smoke 只能走 `s2_0_pipeline_debug`，不得挤占 matrix。

显存/分批：`S2PhysicsEnv` 当前是单环境适配器；实测单进程楼梯运行 GPU 约 1.5 GiB（`1501 MiB / 8192 MiB`），远低于 7168 MiB cap。因此**不需要也不应**按 20 env 并行；150+6 条应串行单环境执行。按本次楼梯约 13 steps/s 的实测量级，419k 步预计约 7–9 小时；H 约 5–10 分钟。实际以 job 日志为准。

## 6. T11 同进程配对与可恢复矩阵运行器

### 6.1 同进程配对（`scripts/run_a2_v09_s2_pair.py`）

同一 Python 进程顺序构建 real/plane env，plane 逐帧接收 real 的动作序列，每帧写 `sim_state_sha256 = sha256(qpos,qvel)`。full 验证 attempt：`t11-pair-stairs-full-r1`。

```text
real_steps=1608
plane_steps=350（plane 在 trigger 前已 nonfoot 终止）
prefix_end_frame=350（未到 map 替换；plane map trigger=None）
prefix_exact=false
first_state_hash_divergence=step 1
branch_prefix_sha256.real=61b67a4bb8369e2d9c92d308af060102b34125d448947fe677ad4c7488c3aa30
branch_prefix_sha256.plane=8646c71250b8d9411c5f448ea2ae24d152b793d986e4a08125f57f57acd9ab0e
real_actions_sha256=a79da2df...
```

结论：**病例/动作前缀一致，但 qpos/qvel 从 step1 起不同**。同进程顺序重建 env 也不可复现同一物理状态；因此不能声明 Q016 通过，也不能把 1e-3 类容差当同态证据。**状态快照/恢复尚未实现，也未验证**。pair summary 明确写 `snapshot_taken=false, snapshot_restored=false, snapshot_status='not_implemented', restored_state_sha256=null, divergence_after_restore=null`。full pair 的 plane 在 350 步 nonfoot 终止，未到触发帧 1528，因此也不能作为“触发时刻同帧”证据。可行替代是完整引擎状态克隆或单 env 双输入/双策略 fork；当前均未实现。

### 6.1b Q024 状态

Q024 要求同一物理快照、同一采样帧的 real/plane 双输入。T11 pair 在 step1 已 qpos/qvel 分叉，因此当前**不满足 Q024 同态输入**；任何 plane-map 结论只能标为“动作/命令前缀一致 + 不同物理状态下的干预对照”，不能标为“只改输入”。在完整状态克隆/同 env fork 实现前，Q024 记 `not_proven`。

### 6.2 可恢复矩阵运行器（`scripts/run_a2_v09_s2_matrix.py`）

- 156 条：S2_DIAG 150 + H 6，串行单环境；
- 每例独立 attempt/证据目录；
- `--resume` 扫描已有 summary，校验 `status=completed + case_sha256 + evidence_files SHA`，通过则跳过；
- 失败写 progress 并继续；
- 每例子进程走既有 ledger reserve/commit/release；runner 自身不写 ledger；
- 每例一行 progress JSONL，结束输出汇总；
- `--limit/--only-task/--dry-run` 已实现。

验证：

```text
limit-2 matrix（t11-limit2）:
  real completed 1609, plane completed 1568, total 3177
  证据: logs/a2_v09/s2_0/matrix/s2-030-real-0140473c4282/r1
        logs/a2_v09/s2_0/matrix/s2-031-plane_diagnostic-0140473c4282/r1

resume check（t11-resume-check）:
  2/2 skipped_completed, total_actual_committed=0, no ledger write

limit-10 dry-run:
  10 条 = stairs 0.3/0.5 × 宽 0.8 的前 5 对 real+plane,
  上限步数 66,728
```

正式剩余预算：

```text
matrix spent=3177, available=417213
剩余 150 矩阵中尚未跑的 148 条上限约 416225
余量 = 988 步
H 六例 = 7874 / 9432
```

整批命令见第 5 节；任何重跑只能走 debug 义务，不能挤占 matrix。

### 6.3 `post_reset`

新代码每帧写 `post_reset`；summary 写 `terminal_row_is_post_reset`。旧 r4/plane 证据不修改 trace，原 SHA 保留，另写：

```text
logs/a2_v09/s2_0/evidence/t10-stairs-real-r4/post_reset.json
logs/a2_v09/s2_0/evidence/t10-stairs-plane-r1/post_reset.json
```

r4 仅 row 1607=true；plane 仅 row 1568=true。Q030 重算已剔除这些行。

### 6.4 Q030 报告口径

见第 4.6 节；Q032 real-r4 top 45 帧全低头均值 -25.6311°，plane-r1 front 18 帧全抬头均值 +14.1021°。旧 Q030 44 帧/-25.96° 与 `positive_forward_lean_fraction` 产物保留但标 superseded。

## 7. 用户观察与分支复现（单例）

- real-r4：40 级爬完、四足入台、top_arrived；平台后持续低头/前倾（Q030: pitch<0），随后 body failure。该阶段和姿态方向与用户观察一致。
- plane-r1：作为干预对照，在平台前缘附近 nonfoot failure，未 top；表现为抬头（pitch>0）。
- 当前证据**不支持**“仿真与现场相反”；也**不支持**“地图替换导致摔倒”作为因果结论：plane 与 real 的干预前物理状态未对齐（qpos/qvel step1 即不同），Q016 同态输入尚未成立。
- 这是单例诊断；未触发奖励、损失、课程或配方变化。

## 8. Q031 同帧双输入反事实（代码已就绪，GPU 采集待整批后）

### 8.1 real 输入采集字段

`--save-inputs`（或矩阵 runner 的 `--save-inputs`）在 real 分支每个控制步、推进物理前保存：

```text
proprio                 [45] float32
history                 [5,45] float32
heightmap               [3,40,30] float32（height/valid/age）
route_error             [3] float32
c_exec                  scalar
recorded_action         [12] float32（同一步真实记录动作）
a_request               [12] float32（输入给动作管理器的动作）
step                    actor step
refresh_index           int
sample_sim_time_s       float
produced_at_actor_step  int
base_sample_z           float
plane_world_z           float（参考顶平台）
refreshed               bool
front_foot_top_support  bool
case_sha256 / weight_sha256=efa9caa8...
```

落盘：

```text
output/inputs.npz
output/inputs_metadata.json
```

`inputs.npz` 不含 reset 出生帧；metadata 记录 `post_reset_excluded=true`。

### 8.2 plane 派生与核验

`src/tasks/velocity/v09/s2_counterfactual.py` 使用正式 `S2PlaneAblation`：

- 事件后首次严格更晚的正常刷新才生效；
- 生效前逐元素完全相同；
- 生效后 `valid/age` 完全不变，只有有效格 `height` 改变；
- 每个 post 帧核验替换值等于 `raw_cache`；
- 保留未触发/无差异病例；
- 任一检查失败 `InputContractError`。

### 8.3 双输入推理与报告

`run_dual_inference` 用同一权重/CPU 推理/归一化，分别对 real 与 plane 输入计算确定性动作均值；不做物理推进。同时：

- real→real 重复推理得数值底噪；
- 若 `recorded_action` 可用，同阶段比较原记录 real 动作；
- 按 `pre_trigger / activation_refresh / post_activation / not_triggered` 汇总 `n`、动作差 RMS、最大绝对差、逐维 RMS/max、动作单位、scale=0.25、`clip_position='not_clipped'`。

报告固定：

```text
evidence_grade = pending
physical_trajectory = not_proven
```

不得由该报告推断“改图导致摔倒或方向改变”。

### 8.4 代表病例缺口

截至本提交，四个代表目录**都缺 `inputs.npz`**（旧 trace 只有状态/动作，不是完整输入）：

```text
t10-stairs-real-r4
t10-stairs-plane-r1
limit-2 real
limit-2 plane
```

缺口清单：

```text
logs/a2_v09/s2_0/evidence/q031_input_coverage.json
```

因此当前只能完成代码与 CPU 测试；Q031 的 pending 证据必须等 GPU 采集。

### 8.5 GPU 采集命令（整批结束后，经批准再执行）

```bash
python scripts/run_a2_v09_s2_0.py \
  --execute --smoke --suite S2_DIAG \
  --case-sha 0140473c4282e721cceec4f0fcd05536337869b9b0379c9947a4059224012150 \
  --branch real --steps 2000 --save-inputs \
  --device cuda:0 \
  --attempt-id q031-stairs-real-inputs-r1 \
  --obligation-id s2_0_pipeline_debug \
  --ledger logs/a2_v09/s2/ledger.jsonl
```

采集后离线：

```bash
python scripts/run_a2_v09_s2_counterfactual.py \
  --inputs logs/a2_v09/s2_0/evidence/q031-stairs-real-inputs-r1/inputs.npz \
  --metadata logs/a2_v09/s2_0/evidence/q031-stairs-real-inputs-r1/inputs_metadata.json \
  --task stairs_walk --device cpu \
  --output logs/a2_v09/s2_0/evidence/q031-stairs-real-inputs-r1/counterfactual_report.json
```

矩阵正式跑时若需覆盖更多 real 病例，加 `--save-inputs`；该参数只对 real 分支生效。

Q032 边界抽核（real-r4，raw trace）：

```text
front: row1545 route 7.9801 (below 8.0), row1546 8.0015 (first), row1547 8.0245
four : row1558 flags [T,T,F,T], row1559 [T,T,T,T] (first), row1560 [T,T,T,T]
top  : row1561 top=False, row1562 top=True (first), row1563 top=True
```

Q030→Q032 对应关系：事件锁存步 1560/1563 比首真 trace 行 1559/1562 大 1；新窗口以首真行为准，不再全局 `step-1`。
