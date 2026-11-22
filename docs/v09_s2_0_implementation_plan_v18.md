# A2 v0.9 证据增量落盘修复（v18）

**状态：** 纯 CPU 测试；未跑 GPU；台账未写；未 git 写
**Supersedes：** `docs/v09_s2_0_implementation_plan_v17.md`（SHA256 `8a8a379ebda66bc1c3bdb110a74583d0d86f896037f5d28f029c79a6aa00b313`）

## 1. 修复

- `S2EvidenceRecorder`：
  - 固定字段仍为 `(T,...)` ndarray；
  - `contact_pairs`/`native_sample_id` 为 object 数组，不再强转同构 ndarray；
  - `trace.jsonl` 每帧一行、变长字段自然支持；写入前递归 `_jsonable` 转换 numpy/torch 标量。
- `attach_trace(path)`：写盘时逐帧 `flush + fsync`；物理步完成即落盘，后续汇总失败不丢已走帧。
- `write()` 同时保留 NPZ + `_trace.jsonl`，并在 metadata 记 `trace_path`。
- smoke 默认证据目录 `logs/a2_v09/s2_0/evidence/<attempt_id>/`，summary 写 `evidence_dir`、`trace_path`、`evidence_frames`、`evidence_partial`。
- summary 增加 `first_64_summary`：pitch min/max、平均速度、平均 `c_exec`、平均/最后支撑足数。

## 2. 测试

`pytest -q tests/test_v09_s2_0_*.py` → **191 passed**。
新增：64 帧变长 `contact_pairs`（0~3 对/帧）完整落盘/读回；后续非法帧失败后已写 64 行不丢。

## 3. 文件/SHA

- `s2_env.py` `ec211727…`
- `scripts/run_a2_v09_s2_0.py` `62af6ed9…`
- `tests/test_v09_s2_0_env.py` `58483d6d…`

## 4. 重跑

```bash
python scripts/run_a2_v09_s2_0.py --execute --smoke --case 0 --steps 64 \
  --device cuda:0 --ledger logs/a2_v09/s2/ledger.jsonl
```
默认新 attempt；证据目录按 attempt 分出；失败仍保留 trace/summary。
未执行：GPU、训练、仿真、env.step、git 写。
