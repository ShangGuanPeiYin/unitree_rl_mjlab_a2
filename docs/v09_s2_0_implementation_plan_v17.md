# A2 v0.9 smoke 足接触聚合修复（v17）

**状态：** 纯 CPU 测试；未跑 GPU；台账未写；未 git 写
**Supersedes：** `docs/v09_s2_0_implementation_plan_v16.md`（SHA256 `cae997ed401c844aaec402c078e461db91534085a45dae71ee39eebe975ea306`）

## 1. 修复

- 新增 `aggregate_foot_contacts(top_forces[4,207], ...)`：
  - 按 v07 已按足索引的 `contacts.top_forces` 对 geom 轴求和，得 `(4,)`；
  - 仅在真实 `top_forces>0` 的 foot/terrain geom 对上构造 `contact_pairs`；
  - 每对含 foot、geom 名称、法向力、时长、足位、surface、valid；
  - 断言 `(4,)`、有限值、无特权接触位。
- `nonfoot_contact_from_links`：排除四足 link，任一非足 body link 力 >10N 即 nonfoot。
- `BackendStep.nonfoot_contact`；`diagnostics`/evidence 落盘。
- 其他形状核对：`foot_velocity_xy` 为 `(4,2)`、`map_valid` 为 `(40,30)`、`foot_contact` 为 `(4,)`，未再使用 207 维原始向量。

## 2. 测试

`pytest -q tests/test_v09_s2_0_*.py` → **190 passed**。
新增：207 维→4 足聚合（无接触=0、单足多接触点求和、非足分类）。

## 3. 文件/SHA

- `s2_physics.py` `04cf7ade…`
- `tests/test_v09_s2_0_physics.py` `1c214027…`

## 4. 重跑

```bash
python scripts/run_a2_v09_s2_0.py --execute --smoke --case 0 --steps 64 \
  --device cuda:0 --ledger logs/a2_v09/s2/ledger.jsonl
```
默认新 attempt；失败按实际步结算、release 未用。台账未改。
未执行：GPU、训练、仿真、env.step、git 写。
