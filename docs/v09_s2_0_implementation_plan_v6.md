# A2 v0.9 T4 原子预留能力与 V5 修正（v6）

**状态：** 纯 CPU；未训练、未仿真、未写正式 S2 台账、未 Git 写
**Supersedes：** `docs/v09_s2_0_implementation_plan_v5.md`（SHA256 `764c3cc8ccdefdad069a633639d863c4dfcc8cf5567db513d1942991e2ac0068`）
**裁定：** DSH-Q015～Q027；Q028/Q029 仅报告字段口径待定，不阻塞
**实现子 agent：** A2-S2-IMPL

---

## 1. V5 阻断修正

| 项 | 落实 |
|---|---|
| B-V5-1 | `s2_state` docstring/常量说明改为 Q027：0.5 s 为完整 0→1 ramp、无额外保持、非 provisional；保留 `q027_ramp_total_0p5s` 快照标识 |
| B-V5-2 | H 六例改为 `legacy_stairs` 适配：不使用 S2 `D_post` 成功判据；仅接收 v07 `v07_success`，硬时限用原件 1301/1302/1306/1315/1318/1332；新增 `legacy_v07_success`（支撑且机身 ≥前缘+0.8 m）；2.0 不成功、2.6 成功回归；六例适配器逐例校验时限 |
| B-V5-3 | S2.0 dry-run 输出 30 个冻结批（两分支×15 组，每批 5 例、含批上界）；`--max-envs` 限制 1..20，200 被 argc 拒绝 |
| 证据自证 | `case_record` 写入 `config_sha256`、`pairing_sha256`、`shared_identity_sha256` |

---

## 2. T4：S2 原子预留写入能力

### 2.1 `s2_ledger.reserve_stage`

- `plan` 校验：stage=S2、SDS=v1.2、items/learning_items、逐项名称/类别/控制步/依据、`total_control_samples`、`learning_total_control_samples`、`stage_total_control_samples`、`physics_substeps=4×stage_total`、billing、admitted、`plan_sha256`。
- 非学习 items 只允许 engineering/evaluation；学习额度单列 learning；`sum(items)=nonlearning envelope`，`items+learning=stage_total`。
- `envelope=total_control_samples ≤ S2_NONLEARNING_CAP`；学习 ≤ 3.6e9；stage ≤ 4e9；S2.0 子额度 ≤ 50,000,000。
- obligation id/name 去重；`plan_sha256` 与内容一致；一次原子预留，重复拒绝。
- 记录每个 obligation 的 `limit/spent/reserved`；`reserve/commit` 按义务结算并超限拒绝；预留不计消费。

### 2.2 现场重算计划（`s2_budget.build_reservation_plan`）

| obligation | 类别 | 控制样本 | 依据 |
|---|---|---:|---|
| resource_probe_1024 | engineering | 3,709,184 | §2.3 |
| resource_probe_512 | engineering | 2,332,928 | §2.3 |
| s2_0_matrix_two_branches | engineering | 420,390 | §4.3/Q020 |
| h_series_baseline_supplement | engineering | 9,432 | §4.3/Q018 |
| h_series_candidate_retest_3_models | evaluation | 28,296 | §4.3 |
| l40_milestone_cache_gate_A | evaluation | 219,220 | §6.1.1/Q021 |
| l40_milestone_cache_gate_B | evaluation | 219,220 | §6.1.1/Q021 |
| gate_admissions_up_to_40 | evaluation | 173,940,000 | §2.5/§6.1.1 |
| final_sets_3_with_frozen_comparisons | evaluation | 83,009,700 | §6.5 |
| waiting_empty_steps_no_extra_charge | engineering | 0（已含批上界，不重复） | §2.5 |
| learning_S2_1 | learning | 800,000,000 | §2.5/§6.2 |
| learning_S2_2 | learning | 1,200,000,000 | §2.5/§6.3 |
| learning_S2_3 | learning | 1,200,000,000 | §2.5/§6.4 |

- **非学习 envelope 合计 263,888,370 ≤ 4 亿**；学习额度 3,200,000,000；stage total 3,463,888,370，受阶段总上限 4e9 约束。
- S2.0 子额度 420,390 + 9,432 = 429,822 ≤ 5,000 万。
- 所有数字由冻结 `S2_DIAG.json`/`h_series_case0.json` 与 `s2_budget` 现场重算，不硬编码 263,888,370。
- `plan_sha256` 当前 `773b53c6…`（每次代码/清单变更会重算）。

### 2.3 `scripts/reserve_a2_v09_s2.py`

- `--dry-run`：只打印 plan 表与合计，不创建/触碰 ledger（测试验证文件不存在）。
- `--write`：仅在显式指定时实例化 `S2Ledger`（默认 `logs/a2_v09/s2/ledger.jsonl`），写前检查无预留、写后打印 receipt（sequence、event SHA、ledger file SHA）与 obligations；重复运行返回 2 并说明已预留。
- 脚本从不触碰 S1 台账；T4 未执行 `--write`，正式 S2 台账仍未创建。

---

## 3. 测试与 dry-run

```bash
pytest -q tests/test_v09_s2_0_*.py
```

**155 passed。** 新增：plan SHA 篡改、总额/重复项/超 cap/子额度超限、重复预留、dry-run 不落盘、`--write` 幂等拒绝、state obligation limits、S1 路径隔离、B-V5-2 2.0/2.6 与六例时限、B-V5-3 30 批与 `--max-envs` 拒绝。

S2.0 dry-run：150 分支 / 30 批 / 75 对 / H 6；`physics_started=false`。
S2.1 train dry-run：1024、49,152 样本/rollout、rollouts `{16276,24414,24414}`；`training_started=false`。

---

## 4. 未决/后续

- Q028（占空比/最长关闭口径）、Q029（四个 g 时刻相对延迟是否全列）只影响报告字段，不阻塞；当前字段已存在，待裁定后调整。
- 正式 S2 预留尚未写入；`q021_reservation_written=false`；需验收组独立核 plan 数字与写入路径后由主对话执行。
- 仍需 GPU：物理/接触/延迟噪声、真实轨迹、分支前缀运行时证明、g_level 真实地形、H 足位补采、PPO/std/Adam/KL、资源峰值/恢复检查。
- 未执行：训练、仿真、env.step、GPU、梯度、Git 写。
