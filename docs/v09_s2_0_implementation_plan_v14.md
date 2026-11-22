# A2 v0.9 smoke 第二次修复（v14）

**状态：** 纯 CPU 测试；未推进物理；未改台账；未 git 写
**Supersedes：** `docs/v09_s2_0_implementation_plan_v13.md`（SHA256 `18a51a58680afdead0f6ef8efa6766951715d1758ae6f17936569a5259487f67`）

## 1. 修复

- `sim_time` 用 `_scalar` 兼容 `TorchArray`/tensor/np 标量/向量：`.item()`、`.detach().cpu()`、数组首元素；多环境取 env 0。
- 终止合同：v07 的 `env_cfg` 原本就把 manager terms 清空，物理终止在 `events.evaluate`（failure masks）。现在构造后 `assert_termination_contract` 断言：
  - manager terms 与 v07 一致（空）；
  - `events.FAILURE_NAMES` 覆盖 body/nonfoot/posture/fall/bounds/contact_overflow/nonfinite；
  - 缺失即拒绝运行。
- `BackendStep.termination_reason` 从 `last_termination` reason code 映射（running/timeout/success/failure names）；写入证据。
- `--preflight`：需正式台账；构造 env、打印 termination 合同、地形 geom 数、机器人关节数、观测形状、显存峰值；`stepped=false`，不 step。
- smoke 路径：Actor 均值 callable；单环境 `observation()`/step 适配；失败保留部分证据并 commit 已推进步。

## 2. 文件/SHA

- `s2_physics.py` `39f01a04…`
- `scripts/run_a2_v09_s2_0.py` `18999bb3…`
- `tests/test_v09_s2_0_physics.py` `3bb6edae…`
- `s2_env.py` `214aef8f…`

## 3. 测试

`pytest -q tests/test_v09_s2_0_*.py` → **187 passed**。
新增 TorchArray 标量、termination 合同、preflight 缺台账拒绝。

## 4. 下一步 preflight（需 GPU 窗口）

```bash
python scripts/run_a2_v09_s2_0.py --preflight --case 0 \
  --device cuda:0 --ledger logs/a2_v09/s2/ledger.jsonl
```

预期输出：termination 合同（manager=0、7 类 failure mask）、geom 数、12 关节、观测形状、peak MiB、stepped=false。
CPU 不能替代真实 mjlab 场景构造，因此需窗口。

## 5. 仍可能失败

S2AtlasCfg/index_geometry 对 20 m/4 m 行；patched map_cache 10Hz；top_forces/foot_lookup；H legacy events；Actor 输入形状。
未执行：训练、仿真、env.step、GPU、梯度、git 写。
