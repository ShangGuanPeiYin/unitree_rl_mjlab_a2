# W0 冻结补充 1：并入 Q001 裁决（v0.6）

- 时间：2026-11-16T15:12+08:00；由 DSH 主对话登记。
- 生效合同 = `doc/sds/SDS_A2_IdealHeightMap_Stairs40_v0.6.md`（`1ed4bf5f…6ad60`，970 行）**+** `logs/a2_ideal_stairs_v06/implementation/coordination/Q001_v06_decision.md`（`5f3f2aa3…a4b1`，82 行）。
- 冲突时以 Q001 裁决为准；原 SDS、原 W0 冻结与旧测试证据保留，不追改旧 SHA。`W0_FREEZE.md` 字节未改（其 SHA `d1b51bc7…c1f0` 已被 `spec.py` 断言引用），本补充为独立文件并与 `W0_MANIFEST_v2.json` 一并冻结。

## 1. 裁决要点（实现口径，逐条对应 Q001）

| 编号 | 裁决 | 实现落点 |
|---|---|---|
| Q001-1 | `air_time` 按 **raw=False** 逐帧累计；`landing = raw & filtered & ~prev_filtered & has_contacted_once & (air>0)`；`bonus=landing*(air−0.10)`；`next_air = raw ? 0 : (has ? air+dt : 0)`；`next_has = has | raw`；prev_raw=raw、prev_filtered=filtered；raw 恢复时**无论是否付款都清零**；单帧失联不付款且不清零到旧腾空；slip/clearance 仍用 filtered。单足金额 `0.2*bonus*dt`，外层不再乘 dt | `v06/rewards.py::AirTimeState.update`；`v06/spec.py` 常量不变 |
| Q001-1 测试 | 必须**经状态机**逐帧验证：15×False→+0.0008 且再 True 不重复；True→False→True 全程 0 且 air=0；多次单帧失联后再 15 帧仍 +0.0008；出生 15×False→首次 True 为 0，随后真实腾空才付；True→3×False→True = 0.06 s→−0.00016；0.05 s→−0.0002 仅作纯函数锚点；四足异步/部分 reset/两帧失联/5 N 阈值/state_dict 中途恢复；缺字段或 shape 不符拒绝 | `tests/test_v06_rewards.py`（新增状态机组） |
| Q001-2 | `RewardFrame.foot_surface_height[B,4]` 为必需字段；同一世界 z 基准；同 tile `TerrainSpec` 查询；缺失/非 finite 拒绝；不平滑/不看前方/不用起跳面 | `v06/rewards.py`、`v06/env.py::_reward_frame` |
| Q001-3 | v06 自有世界布局 `(0, tile_id*8.0, 0)`；楼梯 bounds `(-2, N*d+2, -2, 2)`、flat `(-2,22,-4,4)`；first tread edge local_x=0、平台起点 `N*d`；**不得**再写“完全沿用旧 v02 atlas”（旧 frame x=−10 且有 y 偏移），只能称“一致平移” | `v06/spec.py` docstring/`QA_spec_rewards.md` 措辞更正 |
| Q001-4 | q0 按 FL/FR/RL/RR×(hip,thigh,calf)：`(−.1,.9,−1.8, +.1,.9,−1.8, −.1,.9,−1.8, +.1,.9,−1.8)`；软限位 `m=(lo+hi)/2, r=(hi−lo)/2, soft=(m−.9r, m+.9r)`（**当前 `_soft_limits` 已正确，不改实现**）；数值 hip[−.909,.909]、前 thigh[−2.0655,2.8755]、后 thigh[−1.285,3.665]、calf[−2.6585,−.6515]；须与运行时 `robot.data.default_joint_pos` / `soft_joint_pos_limits` 按关节名对照 ≤1e−6 | `v06/spec.py` 注释更正；`tests/test_v06_spec.py` 增加运行时对照 |
| Q001-5 | fresh 输出固定 `logs/a2_ideal_stairs_v06/train_seed42_trial01`、验收文件固定 `logs/a2_ideal_stairs_v06/implementation/V06_IMPLEMENTATION_ACCEPTANCE.md`；允许解析后的等价绝对路径；目录已存在拒绝且不清空；恢复只走 `--resume-run`；不授予 trial02 | `v06/orchestration.py`、`spec.recipe_config`、`tests/test_v06_entrypoints.py`/`test_v06_orchestration.py` |
| Q001-6 | L40 h=d=.20：c=.2→7257、c=.6→2753 作为公式守卫（非查表替代） | `v06/spec.py::_check_frozen_tables` |
| Q001-7 | `spec.DIAGNOSTIC_COLLAPSE_STREAK=2` 显式常量并纳入配置/SHA；仅用于继承的停滞 collapse；`speed/runner` 读取必需常量、无静默 fallback，缺失预检报错；不得扩展到任意诊断异常；flat/H00 保持集计数、最低子级 4 窗口计数独立保存 | `v06/spec.py`、`v06/speed.py`、`v06/runner.py` |

## 2. 不变项

- 奖励权重、PPO、课程预算、门禁阈值、速度、seed、`30000` 上限、D1 offset、22 tile 几何参数均**不因本裁决改变**。
- R1/R2 文件、R2 seed43 停止状态、历史证据不改；不 commit/push。
- 180 passed 属旧代码证据，保留；放行以修正后完整 CPU + 独立复验为准。

## 3. 放行链（不变）

CPU 修正 → 完整 CPU（0 failed）→ 独立复验（同一冻结 SHA，blocking=0）→ GPU smoke → probe → 独立最终工程验收 → 自动新 v06 seed42 训练（F0/H/D/L/V/R，30000 上限）→ 最终三速×100 例与视频。
