# S2.1 门禁病例冻结（v1）

- 时间：2026-11-22
- 规范：SDS v1.2 §2.2、§6.1.1、§6.2；DSH-Q033
- 冻结工具：`scripts/freeze_a2_v09_s2_gate_cases.py`（append-only，拒绝覆盖）
- 目录：`logs/a2_v09/s2/gate_cases/v1/`（本机证据，不入库）

| 套别 | manifest SHA256 | 病例数 | 控制样本上界 | 文件 SHA256 |
| --- | --- | ---: | ---: | --- |
| S2_GATE_A | `1ea91343afc22ced4c349e2704f828ab21527c6425bc056477e69c42242ab69a` | 720 | 1,620,000 | `98842a079945b55180b772c2809dc7bfd610b8581b5348dbfe38a5ceedc3448f` |
| S2_GATE_B | `9af3f7c93b613dc24c2a02c2296594c0cf92efff7adb4c1880766881cf38d2ff` | 720 | 1,620,000 | `ecdbf5c42a31528f80eedbee141a763689685948b944cfa3557068cc23093056` |

- 组成：S2.1 36 组 × 20 例 = 720 例/套（平地 9、stairs_walk 10/20 级×宽 2×速 3 固定 5 m 12、platform_walk 顶高 2/4 m×宽 2×速 3 固定 5 m 12、旧 L40 里程碑 3）。
- 预算：A+B = 3,240,000 控制样本，占 `gate_admissions_up_to_40`（173,940,000）。
- 门禁执行器：`scripts/run_a2_v09_s2_gate.py`，`--suite gate_A|gate_B --course S2.1 --checkpoint <schema11> [--dry-run|--execute]`，只读冻结清单。
- 门禁时点：S2.1 训满 8 亿样本（32,552 rollouts）或按 SDS 提前达标后执行；两组 ≥18/20 且平地保持与逐例质量通过，连续两次门禁通过才进 S2.2。
