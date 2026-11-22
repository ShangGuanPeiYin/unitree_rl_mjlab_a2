# A2 v0.9 smoke attempt_id 修复（v16）

**状态：** 纯 CPU 测试；未跑 GPU；台账只追加；未 git 写
**Supersedes：** `docs/v09_s2_0_implementation_plan_v15.md`（SHA256 `7363464aac8bb96636cd9932d74eacee5cfbf213e716ddcf105fafdfb1b61f43`）

## 1. attempt_id

- `s2_execute.new_attempt_id()`：UTC 时间戳 + 6 位短随机。
- `S2ExecutionDriver(..., attempt_id=...)`；操作 id 变为 `{branch}:{group}:{seq}:{attempt}`。
- `run_a2_v09_s2_0.py --attempt-id` 可选；默认新 id；禁止 `:`/`/`。
- smoke 操作 id `smoke:{case}:{attempt}`；preflight 不 reserve 但打印同一 attempt。
- 证据 `case_record`/driver 记录 `attempt_id`；smoke summary 记录。

## 2. 重试语义

- 同 attempt 的同一 operation 重复 reserve 仍拒绝（台账 operation_id 全局唯一）。
- 失败/释放/退款关闭后，新 attempt 用新 id 合法。
- 历史 `smoke:0` reserve/commit/refund 事件保留；新尝试使用 `smoke:0:<attempt>`。

## 3. 测试

`pytest -q tests/test_v09_s2_0_*.py` → **189 passed**。
新增：同 attempt 重复拒绝、不同 attempt 允许、关闭后新 attempt、driver 记录 attempt_id、CLI 非法 attempt 拒绝。

## 4. 文件/SHA

- `s2_execute.py` `5cbb1f79…`
- `s2_env.py` `d55e69ea…`
- `scripts/run_a2_v09_s2_0.py` `c3ed4fc6…`
- `tests/test_v09_s2_0_execute.py` `94cc3ca0…`
- 台账 SHA `bf500771…`（未改）。

## 5. 重跑

```bash
python scripts/run_a2_v09_s2_0.py --execute --smoke --case 0 --steps 64 \
  --device cuda:0 --ledger logs/a2_v09/s2/ledger.jsonl
```
默认自动生成新 attempt；也可显式 `--attempt-id smoke-r2`。
未执行：GPU、训练、仿真、env.step、git 写。
