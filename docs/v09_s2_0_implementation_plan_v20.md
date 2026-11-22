# A2 v0.9 观测诊断与 S1 参考轨迹（v20）

**状态：** 纯 CPU 测试；未跑 GPU；台账未写；未 git 写
**Supersedes：** `docs/v09_s2_0_implementation_plan_v19.md`（SHA256 `88949d58a70663bf48db633240b454dd56e5848530ed2dc3d92689cb5a098885`）

## 1. 观测诊断

- `s2_env.observation_diagnostics`：逐 key 输出 dtype/device/shape/expected/min/max/NaN/Inf 计数、非有限索引（最多20）。
- `validate_policy_observation`：shape/dtype/有限性失败时 `ObservationContractError` 附带完整 diagnostics，指出 `_obs` key 与具体维度。
- 真实 backend `_observations()` 已 `squeeze` 首维 `[1,...] -> [...]`；`S2PhysicsEnv._policy_observation` 再做通用 batch 维剥离。
- 首步：reset 后 `_observations()` 提供 joint_vel/action 零值；若仍有非有限/空 key，dump 会直接显示。

## 2. `--dump-observation`

```bash
python scripts/run_a2_v09_s2_0.py --dump-observation --case 0 \
  --device cuda:0 --ledger logs/a2_v09/s2/ledger.jsonl
```
只 reset 不 step；写 `logs/a2_v09/s2_0/evidence/<attempt>/observation_dump.json`，含 dtype/shape/min/max/NaN/Inf/索引与设备信息。

## 3. S1 参考轨迹

`scripts/compare_a2_v09_s2_observation_parity.py --run-s1-reference`：
- `S1_DIAG` flat_walk case0，`V09S1Env([case], device, seed=42)` + `s1_least.pt` Actor 确定性均值，64 步；
- 输出 `logs/a2_v09/s2_0/parity/s1_flat_case0/trace.jsonl`；
- 不写 S1 台账；打印 S2 预留参数（`s2_0_matrix_two_branches`, samples=64），由主对话统一 reserve/commit。
命令：
```bash
python scripts/compare_a2_v09_s2_observation_parity.py --run-s1-reference \
  --case 0 --steps 64 --device cuda:0 \
  --output logs/a2_v09/s2_0/parity/s1_flat_case0/trace.jsonl
```
现有 `evaluate_a2_v09_s1.py` 不能复用（强制 S1 台账/身份），故此模式替代。

## 4. 测试

`pytest -q tests/test_v09_s2_0_*.py` → **195 passed**。
新增：非有限 proprio 定位报错、dtype/shape 报错、观测契约 diagnostics、S1 参考模式静态入口。

## 5. 文件/SHA

- `s2_env.py` `13a3676e…`
- `s2_physics.py` `ae01766c…`
- `scripts/run_a2_v09_s2_0.py` `b03b3e74…`
- `scripts/compare_a2_v09_s2_observation_parity.py` `d13f8599…`
- `tests/test_v09_s2_0_env.py` `c2cdf7aa…`

## 6. 重跑顺序

1. `--dump-observation` 看真实 dtype/shape/NaN。
2. 修完观测后 64 步 smoke。
3. 同窗口跑 `--run-s1-reference`，对照两条 trace。

未执行：GPU、训练、仿真、env.step、git 写。
