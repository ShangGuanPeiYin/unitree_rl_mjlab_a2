# A2 v0.9 smoke run_id 修复（v13）

**状态：** 纯 CPU 测试；正式台账只读；未推进物理；未 git 写
**Supersedes：** `docs/v09_s2_0_implementation_plan_v12.md`（SHA256 `19040504e88501002eedb00d19dbd6e2f68527edf6e2bd3264c2eccf2ba73569`）

## 1. 修复

- `s2_ledger.read_run_id(path)`：只读 header，不写台账。
- `run_a2_v09_s2_0.py`：`--run-id` 可选；默认读取 header；显式值与 header 不一致时在台账构造/GPU/Actor 前 `SystemExit`。
- `s2_execute.ReservationGate` 移除硬编码 `run_id`。
- `reserve_a2_v09_s2.py`：已有台账用 header run_id；显式不一致先拒绝；新台账才用默认 `a2-v09-s2-q021-r1`。
- 其他入口检查：`evaluate_a2_v09_s2_0.py`、`train_a2_v09_s2.py`、`s2_resources.py` 不使用台账 run_id，无需改。

## 2. 回归

- `read_run_id` 读回正式台账 header `a2-v09-s2-q021-r1`。
- 临时台账 header `reserved-run` + 执行参数 `wrong-run` → 在任何物理/证据前 `SystemExit`，无 smoke 输出。
- 一致时 `S2Ledger` 正常构造（既有测试）。

## 3. 测试

`pytest -q tests/test_v09_s2_0_*.py` → **185 passed**。

## 4. 文件/SHA

- `s2_ledger.py` `2a8f2d90…`
- `s2_execute.py` `c256e898…`
- `scripts/run_a2_v09_s2_0.py` `cd59c844…`
- `scripts/reserve_a2_v09_s2.py` `a425dfea…`
- `tests/test_v09_s2_0_reserve.py` `e6c5aebe…`

## 5. 正式台账

`logs/a2_v09/s2/ledger.jsonl` 未清空/未重写；header run_id `a2-v09-s2-q021-r1`，plan `773b53c6…`，envelope 263,888,370。

重跑 smoke：
```bash
python scripts/run_a2_v09_s2_0.py --execute --smoke --case 0 --steps 64 \
  --device cuda:0 --ledger logs/a2_v09/s2/ledger.jsonl
```
未执行：GPU smoke、训练、仿真、env.step、git 写。
