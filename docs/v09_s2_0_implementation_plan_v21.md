# A2 v0.9 observation dump 实测（v21）

**状态：** 已跑 1 次 `--dump-observation`（只 reset，不 step，0 控制步）；未训练/未 git 写
**Supersedes：** `docs/v09_s2_0_implementation_plan_v20.md`（SHA256 `02ee052c1f248d4d61f75ca4ec388e56dd1c6ab10c1aa9c969a585478b77a223`）

## 1. 诊断收口

- `jsonable` 成为唯一递归序列化入口：torch（含 0 维）、TorchArray/warp、numpy（含 object）、dict/list/tuple、标量/None；未知对象抛带类型与路径的 TypeError。
- `observation_diagnostics` 不再 `np.asarray(value)`；dtype 取 `to_numpy` 后数组，设备独立记录。
- `validate_policy_observation` 的异常上下文走 `jsonable`；trace/npz metadata/summary/parity 均走 CPU/native。
- 边界函数源码测试禁止 `np.asarray(value)`；torch 张量 diagnostics/写盘测试通过。

## 2. GPU dump 实测

命令：
```bash
python scripts/run_a2_v09_s2_0.py --dump-observation --case 0 \
  --device cuda:0 --attempt-id dump-r3 \
  --ledger logs/a2_v09/s2/ledger.jsonl
```
输出：`logs/a2_v09/s2_0/evidence/dump-r3/observation_dump.json`，`status=ok`，`stepped=false`，未 reserve/未消费。

### proprio (dtype float32, shape [45], NaN=0, Inf=0)
```text
[0,0,0, 0,0,-1, 0.3,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,
 0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0]
min=-1.0 max=0.3000000119
```
- dims 0–2 ω_b=0；3–5 gravity=[0,0,-1]；6–8 command=[0.3,0,0]；9–20 q−q0=0；21–32 qdot=0；33–44 last action=0（首帧正确）。
- history: float32 [5,45], NaN/Inf=0
- heightmap: float32 [3,40,30], NaN/Inf=0, min=-0.2666667 max=1.0
- route_error: [-0.1925202, 0.0121768, 0.9999259]

## 3. S1 参考命令

```bash
python scripts/compare_a2_v09_s2_observation_parity.py --run-s1-reference \
  --case 0 --steps 64 --device cuda:0 \
  --output logs/a2_v09/s2_0/parity/s1_flat_case0/trace.jsonl
```
不在 S1 台账写入；S2 预留参数由主对话处理。

## 4. 测试

`pytest -q tests/test_v09_s2_0_*.py` → **197 passed**。

## 5. 文件/SHA

- `s2_env.py`（诊断/jsonable）见当前 SHA
- `scripts/run_a2_v09_s2_0.py`
- `tests/test_v09_s2_0_env.py`
- 本文件

## 6. 下一步

观测首帧已确认有限、形状正确、command=0.3。可直接 64 步 smoke；同窗口跑 S1 参考后对照两条 trace。
未执行：训练、仿真步进、git 写。
