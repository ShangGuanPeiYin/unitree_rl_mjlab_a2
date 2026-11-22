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
s2_0_matrix_two_branches: limit 420390, spent 384, available 420006
s2_0_pipeline_debug:      limit 20000, spent 7682, available 12318
h_series_baseline_supplement: limit 9432, spent 0, available 9432
```

## 3. 同病例 S1/S2 parity（B-V8-3）

旧 r9 parity 的 S1 是 `S1_DIAG` flat case0，S2 是 `S2_DIAG` flat case0；spawn/streams 不同，不能作同病例轨迹对照。

新增 `scripts/run_a2_v09_s2_parity.py`：从 S2 病例复制 spawn/yaw/streams/command，构造 `V09S1Env` 的 **flat_walk flat tile 0** 代理；S1 与 S2 接收同一 S1 Actor 动作序列，逐帧记录观测 SHA256 与逐键最大差。

flat case 结果（`aee59c5d3a49...`，64 步）：

```text
s2_case_sha256 = aee59c5d3a49a300c85a0a0067841059d36ea283ce1c0e62589477444e8bcef2
s1_proxy_case_sha256 = e3887ccdb0503b76f9fac2d4b88ba854b0ec34cbae9426d5be5ed6220b0b9d2e
spawn = (-1.0858503909, -0.0385040479)
route 最大差 = 1.788e-7 m
obs 最大差: proprio 1.746e-4, history 1.746e-4,
            heightmap 5.96e-8, route_error 1.624e-6
practical parity (1e-3) = true
strict (1e-6) = false（GPU float 噪声）
```

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
pitch after top: mean -25.38°, min -53.19°, positive(前倾) 0%
四足每足最大高度约 8.12–8.25 m，四足均完成抬升
```

判断：确实上梯并到达平台顶部；平台后没有前倾，而是持续抬头/后仰（pitch 变负），在约 route 9.27 m 处 body failure。

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
pitch after platform: mean +13.36°, max +15.91°, positive 94.7%
```

plane 在平台前缘附近（约 8.15 m）失败，未锁存 top；平台后呈明显前倾，与 real 的后仰方向相反。

### 4.3 分支前缀

normal real/plane 是独立 GPU 进程；同一动作序列 replay（`t10-stairs-plane-replay300`）证明 `a_request`/`a_applied` 逐帧完全一致，但 joint-level 状态在约 step 5 后出现 GPU solver order 数值漂移（route 5e-4 m，feet 4e-3 m，joint velocity 最大 0.73），因此**不能声称跨进程 trace 逐位相同**。这是数值执行限制，不是 map branch 输入不同；详见：

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

预算：S2 矩阵义务 available 420,006，419,402 后可余 604；H 义务 9,432，可余 1,558。任何重试/额外 smoke 只能走 `s2_0_pipeline_debug`。

显存/分批：`S2PhysicsEnv` 当前是单环境适配器；实测单进程楼梯运行 GPU 约 1.5 GiB（`1501 MiB / 8192 MiB`），远低于 7168 MiB cap。因此**不需要也不应**按 20 env 并行；150+6 条应串行单环境执行。按本次楼梯约 13 steps/s 的实测量级，419k 步预计约 7–9 小时；H 约 5–10 分钟。实际以 job 日志为准。
