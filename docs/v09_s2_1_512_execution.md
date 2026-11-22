# S2.1 512 训练执行准备（V10 修复后）

状态：代码与 CPU 测试已修正；512 independent-copy probe 已完成并通过
（`admitted=true`，SHA `bdb02d67…`）。admission 尚未写入，必须等 V10
重新复核给出 review SHA。

## 1. 512 probe 证据

- 报告：`logs/a2_v09/s2/probe/512-r5/probe_report.json`
- 独立副本证据字段：
  - `entry_weight.source_sha256_before == source_sha256_after`；
  - `entry_weight.probe_copy_sha256 == source_sha256_before`；
  - `entry_state.sha256_before == sha256_after_reload`；
  - `process.pid/ppid/argv/launcher_argv`，`independent_process=true`；
  - `independent_copy=true` 且 `failure_reasons=[]`。
- `run_probe` 只有在上述证据全部通过且调用方显式传入
  `--independent-copy` 时才写 `independent_copy=true`；否则加入
  `failure_reasons`。
- `write_admission`/`validate_admission` 不再只信 `admitted`，会重算所有
  requirements（peak/host/warmup/measurement/eval/video/restore/
  independent_copy/entry SHA）。

## 2. 512 训练预算

```text
samples_per_rollout = 512 × 48 = 24,576
S2.1 cap             = 800,000,000
max_rollouts         = floor(800,000,000 / 24,576) = 32,552
effective_cap        = 32,552 × 24,576 = 799,997,952
checkpoint cadence   = ceil(5,000,000 / 24,576) = 204 rollouts
```

- 最终检查点：32,552 rollouts / 799,997,952 samples（剩余 2,048 不足一个
  rollout）。
- 1024 路径仍为 16,276 rollouts / 49,152 samples / 102 checkpoint cadence。

## 3. admission 写入

V10 复核给出合法 64-hex review SHA 后，用 `write_admission`（它会在写入前
重新校验 512 报告）：

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
/home/xxs/anaconda3/envs/a2/bin/python - <<'PY'
import json
from pathlib import Path
from src.tasks.velocity.v09 import s2_resources
probe = Path('logs/a2_v09/s2/probe/512-r5/probe_report.json')
report = json.loads(probe.read_text())
doc = s2_resources.write_admission(
    'logs/a2_v09/s2/admission/s2_1_512_r1.json',
    probe_report=report, probe_path=probe, num_envs=512,
    independent_review_sha256='<V10_REVIEW_SHA256>',
    independent_review_path=Path('docs/v09_s2_1_admission_independent_review_v10.md'))
print(json.dumps(doc, ensure_ascii=False, indent=2))
PY
```

## 4. 训练启动命令

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
/home/xxs/anaconda3/envs/a2/bin/python scripts/train_a2_v09_s2.py \
  --execute --num-envs 512 \
  --fallback-evidence logs/a2_v09/s2/probe/1024-r2/probe_report.json \
  --device cuda:0 \
  --ledger logs/a2_v09/s2/ledger.jsonl \
  --probe-report logs/a2_v09/s2/probe/512-r5/probe_report.json \
  --admission-evidence logs/a2_v09/s2/admission/s2_1_512_r1.json \
  --output-dir logs/a2_v09/s2/s2_1 \
  --max-rollouts 32552 \
  --checkpoint-every-rollouts 204 \
  --attempt-id s2.1-r1
```

GPU 前拒绝条件（均已实现）：

1. `--num-envs 512` 缺 `--fallback-evidence`，或 1024 报告不是
   `num_envs=1024 / admitted=false / failure_reasons` 非空；
2. ledger 不存在；
3. `--probe-report` 或 `--admission-evidence` 不存在；
4. 512 probe 报告 `num_envs` 不符或 `admitted=false`；
5. admission schema/num_envs/probe SHA/review SHA/entry SHA/所有 probe
   requirements 任一不符；
6. `learning_S2_1` obligation 不存在、limit 小于 effective_cap，或
   `reserved != 0`。

## 5. 预计时长与 OOM

- probe 50 次测量实测约 **5.58 s/rollout**（512-r5：278.96 s / 50）。
- 完整训练约 `32,552 × 5.58 s ≈ 181,700 s ≈ 50.5 h`；按 5.2–6.0 s/rollout
  给出 **47–54 h** 训练窗口（不含少量检查点写入）。
- 512 probe peak 4,521.75 MiB；训练若 OOM：
  - micro 已是 512，不能再按“先降 micro”处理；
  - `S2_ENV_CANDIDATES` 只有 1024/512，**不能静默降到 256**，也不能改物理、
    地图、观测、网络或奖励；
  - 只能停止并报告，申请新的环境档位/预算裁定，另做 probe/admission。

## 6. resume

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
/home/xxs/anaconda3/envs/a2/bin/python scripts/train_a2_v09_s2.py \
  --execute --num-envs 512 \
  --fallback-evidence logs/a2_v09/s2/probe/1024-r2/probe_report.json \
  --device cuda:0 \
  --ledger logs/a2_v09/s2/ledger.jsonl \
  --probe-report logs/a2_v09/s2/probe/512-r5/probe_report.json \
  --admission-evidence logs/a2_v09/s2/admission/s2_1_512_r1.json \
  --output-dir logs/a2_v09/s2/s2_1 \
  --resume "$(ls -1t logs/a2_v09/s2/s2_1/s2_1_*.pt | head -n1)" \
  --max-rollouts 32552 \
  --checkpoint-every-rollouts 204 \
  --attempt-id s2.1-r2
```

- `resume` 读取 schema 11，恢复 model/optimizer/LR/std/learning_samples/
  rollouts，不重置 LR/std。
- 训练中普通异常：`train_rollout` 的异常路径调用
  `budget.finish(attempt_id, 0)`，只发 `release`，不 commit 本次 rollout；
  已有单测注入异常验证。
- 硬杀/断电不会执行 Python `finally`；若 ledger 仍有 pending reservation，
  CLI 会在 GPU 前拒绝，必须先结算该 reservation 再 resume。resume 使用新的
  `--attempt-id` 避免重复 operation_id。

## 7. 门禁（待补执行器）

- `s2_train.gate_plan()`：36 组、每组 20 例、A/B 两套、阈值 18/20。
- S2.1 一套 36×20 的上界为 1,620,000 控制样本；A/B 两套为 **3,240,000**。
- 计费义务 `gate_admissions_up_to_40` 上限 **173,940,000**（40 次最坏课程
  gate admission），S2.1 A/B 预计 3,240,000 ≤ 173,940,000。
- L40 里程碑使用独立缓存义务
  `l40_milestone_cache_gate_A/B`，每套 219,220，合计 **438,440**；不重复
  计入每次 gate admission。
- **当前没有 S2.1 gate 执行 CLI**：仓库只有 `gate_plan`、
  `evaluate_gate_results` 和预算函数，`evaluate_a2_v09_s2_0.py` 只支持
  `S2_DIAG/H_SERIES`，不能给 36 组 A/B 的真实启动命令。需先补 gate
  executor（或明确授权的新脚本）后才能启动门禁。
