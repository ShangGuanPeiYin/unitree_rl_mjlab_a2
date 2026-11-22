# A2 v0.9 smoke 第三次修复（v15）

**状态：** 纯 CPU 测试；台账已追加冲正；未跑 GPU；未 git 写
**Supersedes：** `docs/v09_s2_0_implementation_plan_v14.md`（SHA256 `7198a02fa8fd479cc440029d3517b9722f46508183c31a59f9cac11bbf0f37b1`）

## 1. native_sample_id

- 真实 backend 构造帧 producer step 用 `max(0, steps-1)`，并强制 `int`；reset 后为 0。
- `assert_native_alignment`：`first_consumed is None` 时不产生 `s−t`，直接通过；有消费者时仍断言 `produced ≤ consumed`、`s−t ≥ 0`。
- fake reset 不推进计数，构造帧 producer=0。

## 2. 计费

`ReservationGate.finish_batch` 改为 §2.5 实际推进：
- `commit = 实际推进步数`；未用部分追加 `obligation_release`，`reserved` 释放；
- 失败批次用实际推进步结算（0 则只 release）；
- smoke CLI 失败时 `actual_samples=len(rows)`，不再用 `steps` 兜底。

真实台账已追加冲正（不改历史行）：
- 事件 sequence 4 `obligation_refund`，`s2_0_matrix_two_branches` 64，operation `smoke:0:refund-failed-construction`，reason 说明构造期失败、撤销错误 commit；
- 当前 ledger SHA `bf500771…`，spent/reserved 全 0；envelope/plan 不变。

## 3. 测试

`pytest -q tests/test_v09_s2_0_*.py` → **187 passed**。
新增/更新：native 构造帧、release/refund、实际计费 commit/release、失败批次结算。

## 4. 文件/SHA

- `s2_physics.py` `70610054…`
- `s2_ledger.py` `7d51c725…`
- `s2_execute.py` `e135bbc3…`
- `scripts/run_a2_v09_s2_0.py` `97ba43e3…`
- `tests/test_v09_s2_0_ops.py` `7ad3937d…`

## 5. 重跑

```bash
python scripts/run_a2_v09_s2_0.py --preflight --case 0 --device cuda:0 \
  --ledger logs/a2_v09/s2/ledger.jsonl
```
preflight 通过后 64 步 smoke；失败即停、实际步结算、release 未用预留。

未执行：GPU、训练、仿真、env.step、git 写。
