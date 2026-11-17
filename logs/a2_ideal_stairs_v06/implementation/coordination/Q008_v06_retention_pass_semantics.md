# Q008：保留集 gate 记录沿用普通门禁阈值（passed 标志错误，行为停止逻辑不受影响）；CUDA/env RNG 快照缺口

- 时间：2026-11-17T06:40+08:00；提出：DSH 主对话。
- 合同：SDS `1ed4bf5f…` + Q001–Q007（Q007 `567df511…`）；Q007 段闭包 `24c7781e…`；Q007 段活动目录 `logs/a2_ideal_stairs_v06/train_seed42_trial02_resume_000400_q007/`。

## 事实

1. Q007 恢复成功：`resume_loaded_000400.pt`（`39e0abd0…`，aligned=true）、`resume_ready_000400.pt`（`c9da2079…`，aligned=true），flat retention 15 例执行一次（failed_cases=0），课程决策一次，`commit_000401`（`a5193573…`）起训练继续，当前约 update 450（H，level_tick=4、generation=1、gate_streak=2；lr≈7.59e-5）。
2. **缺陷**：`gates/flat_retention_000400.json` 中 `gate.passed=false`、`summary.per_speed[i].case_pass_required=18`、`case_pass_count=5/5`、`failed_cases=0`。即保留集记录沿用了普通门禁的“每速 18/20”门槛，而保留集每集只有 15 例（三档各 5），其判定应是“失败数达到 12/15 才计入 regression streak”。因此 `passed` 标志与 `case_pass_required` 对保留集无意义且误导（`flat_retention` 永远显示 false）。
3. **行为影响核查**：`runner._apply_retention` 只用 `failed_cases` 调 `curriculum.record_retention`；后者只用 `failed >= RETENTION_FAIL_MIN(12)` 决定 streak/stop，完全忽略 `passed`。故本次 `passed=false` **未触发停机、未增 streak**，训练继续；仅记录/汇总语义错误。
4. **O-15（v11 独立验收发现）**：`evaluation.training_state_snapshot` 只比较 CPU torch/NumPy RNG，不含 CUDA/env/sampler；实测 `exception_000400` 与 `boundary_000400` 仅 CUDA RNG blob 不同（`a74d…` vs `5419…`）。Q007 恢复保留 exception 自身 CUDA 流、未拼接；SDS 8.3“评估前后比较 RNG”存在覆盖缺口。

## 请求裁决

- **候选 A（建议）**：**不中断当前训练**（缺陷只影响记录，不影响停止/streak），继续到 600（flat+H00 双保持）与后续门禁；把“保留集 `passed`/`case_pass_required` 语义修正”列入训练结束后的修复（或下一版 r3），并在最终报告中把 400 的 `passed=false` 明确标为记录缺陷、以 `failed_cases=0` 为实际判定；同时接受 O-15 为已知观察项（记录 CUDA RNG 差异调查，不改采样）。
- 候选 B：立即修复并再次从 400/450 回退恢复（需新 bridge/段与重新验收；当前 checkpoint 可按 interval/boundary 保存，但会打断训练并新增工程成本）。
- 候选 C：其他处置。

## 建议的具体修复（供 A 采纳后实施）

- `evaluation.summarize`/`gate_result` 对 `kind in ('flat_retention','stair_retention')` 使用保留集判定：`case_pass_required = None` 或 `12/15` 语义明确、`passed = failed_cases < 12`（并记录 `fail_required=12`）；每速 5 例只作明细，不用 18/20。
- 保留 `failed_cases`、case 表 SHA、逐例记录不变；不影响普通门禁 18/20 与最终 95/100。
- 加测试：F0/H/D/L/V/R 下 flat/stair retention 的 `passed`、`fail_required=12`、`failed_cases=0/11/12/15` 边界；普通门禁仍 18/20。

## 边界

- 不改变奖励/PPO/课程/门禁阈值/速度/seed；不新建 fresh；不恢复 R2 seed43；不 commit。
- 在裁决前不主动中断当前训练；若 600 双保持因该记录语义触发误停机，我会保留证据并立即上报。
