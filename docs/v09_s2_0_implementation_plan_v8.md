# A2 v0.9 V6 阻断修复 + T5 执行编排（v8）

**状态：** 纯 CPU；未训练、未仿真、未 GPU、未写 S2 台账、未 Git 写
**Supersedes：** `docs/v09_s2_0_implementation_plan_v7.md`（SHA256 `cf179dec3ccfff8aac69987afe48218c3985ce284df01f1fc4629339136970e6`）
**裁定：** DSH-Q015～Q029

---

## 1. V6 七项阻断

| 项 | 修复 |
|---|---|
| B-V6-1 | `stable_stand` 仅在 DECEL/HOLD 且 `|c_exec|≤1e-7` 且 I_stable 时 +1；0.25/0.1 不付，0/1e-12 付。 |
| B-V6-2 | acceptance 补 `valid_cruise_window`、站立 roll/pitch≤10°、机身高度≥0.30、27组×每组100、9 平地组对原 6400 保持门槛；缺参考/份额不足即失败。 |
| B-V6-3 | 512 仅在 1024 明确失败后可选；train CLI `--num-envs 512` 需 `--fallback-evidence`（1024 admitted=false）；probe 加 1 环境视频；固定 `PROBE_EVAL_TIMEOUT_STEPS=9472`（含末步，9471 为最后零基索引）。 |
| B-V6-4 | S2 台账 operation_id 全局唯一；重复 reserve/commit 直接拒绝。 |
| B-V6-5 | checkpoint 补 loader、schema/SDS 跨版本拒绝、A/B 序列+case/candidate SHA、teacher/entry SHA、model/optimizer 非空。 |
| B-V6-6 | runner 补 entropy、approx KL、逻辑 minibatch 先定 LR 再一次 Adam、教师 eval/normalizer 冻结断言、课程切换不重置 LR/std、S2.1 新 Adam + std 继承。 |
| B-V6-7 | 首米窗 `first_meter_window_mask`；`first_full_interval_start`=连续满足区间起点 t0（向量为 0）；旧报告去掉 duty/max_off；四足/几何事件按控制步号映射，非零起点/缺帧有测试。 |

## 2. T5 执行编排

新增 `s2_execute.py`（`37179824…`）：
- 30 冻结批（两分支×15组×5例）；每批先 `reserve` 批上界，结束后按上界 `commit`（含等待空步/padding），actual 仅作证据。
- `ReservationGate` 拒绝无 stage/无义务/余额不足；`reservation_preflight` 供 CLI。
- `S2ExecutionDriver` 注入 `env_factory`/Actor均值 policy；逐例 latch 首次终止，终止后不再推进；证据经 `s2_env.case_record`（config/pairing/shared SHA、g_level 报告）。
- `device_report` 打印设备/显存/7168 MiB 上限；单进程。

`scripts/run_a2_v09_s2_0.py`（`1da11a48…`）：`--plan-only` 打印 30 批；`--execute --smoke --case <id> --steps 64` 需真实 S2 台账预留；
`build_factory` 从 `src.tasks.velocity.v09.s2_physics.build_s2_env` 取物理装配，缺失时在分配 GPU 前明确报错。
**当前 `s2_physics` 装配尚未提供**，因此真实 GPU smoke 仍被阻断；CPU 侧批表/预留/终止/证据已测。

## 3. 测试

```bash
pytest -q tests/test_v09_s2_0_*.py
```

**173 passed。** 新增 V6 反例（stable 0.25、acceptance 50/50+缺6400、512无证据、重复 operation、跨版本 checkpoint、PPO KL/一次 Adam、首米/非零步映射）与 T5 批表/预留/提交/fake 驱动/CLI 拒绝。

## 4. 预算/预留

plan `773b53c6…` 不变：非学习 263,888,370，学习 3,200,000,000，stage 3,463,888,370，S2.0 429,822；`q021_reservation_written=false`，正式 S2 台账未创建。

## 5. 仍缺

- `s2_physics.build_s2_env` 真实 v09/mjlab 装配 + Actor 均值 policy（GPU smoke 前置）。
- GPU smoke 1 例 64 步、全 150 分支、H 六例 legacy 证据、资源峰值/恢复检查。
- Q028/Q029 字段已实现；Q028 窗口/原因测试通过。
