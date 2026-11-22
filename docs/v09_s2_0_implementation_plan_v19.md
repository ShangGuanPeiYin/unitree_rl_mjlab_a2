# A2 v0.9 CUDA 张量边界修复（v19）

**状态：** 纯 CPU 测试；未跑 GPU；台账未写；未 git 写
**Supersedes：** `docs/v09_s2_0_implementation_plan_v18.md`（SHA256 `889d0e8c99028da999d4503bf44ccbad7404a34e08af9d9382bb6d5f5566ed38`）

## 1. 边界规则

- `s2_env.to_numpy(x)`：统一 `torch.Tensor.detach().cpu().numpy()`、`.numpy()`、numpy/list/标量；证据/统计/写盘/断言只用 CPU numpy/Python。
- **Actor 路径保留 device 张量**：`S2PhysicsEnv._policy_observation` 对 torch 张量原样返回（不 H2D 往返）；只有 fake/ numpy 输入才 `torch.as_tensor(device)`。
- 证据路径：`S2EvidenceRecorder.append` 边界 `to_numpy`；`trace.jsonl`/`_jsonable` 原生类型；`diagnostics()` 边界转换。
- `S2PhysicsEnv.observe` 缓存 device 张量并 `.clone()`。
- parity `_array` 继续 `.detach().cpu()`。

## 2. 修复位置

`observe()`、`_policy_observation`、`diagnostics`、`S2EvidenceRecorder` 数组/JSONL、summary 统计、parity 比较器。

## 3. 测试

`pytest -q tests/test_v09_s2_0_*.py` → **194 passed**。
新增 `to_numpy` tensor/list/scalar、Recorder 接收 torch 张量、边界 numpy 化。

## 4. 文件/SHA

- `s2_env.py` `86f8e7e4…`
- `s2_physics.py` `38dfdef2…`
- `s2_parity.py` `118bcaf1…`（未变）
- `tests/test_v09_s2_0_env.py` `bca0522c…`

## 5. 重跑

```bash
python scripts/run_a2_v09_s2_0.py --execute --smoke --case 0 --steps 64 \
  --device cuda:0 --ledger logs/a2_v09/s2/ledger.jsonl
```
预期：Actor 直接消费 GPU 观测；证据落盘全 CPU/原生；失败按实际步 commit/release。
未执行：GPU、训练、仿真、env.step、git 写。
