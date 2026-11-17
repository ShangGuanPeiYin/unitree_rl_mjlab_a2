# Q001：v0.6 实施中的接口/语义待裁决项（不阻塞 CPU 工作）

- 时间：2026-11-16T15:10+08:00；提出：DSH 主对话（汇总 A 的 `QA_spec_rewards.md` 及集成检查）。
- 依据：SDS v0.6（`1ed4bf5f…6ad60`）、W0 冻结（`d1b51bc7…c1f0`）。
- 状态：A/B/C 已按下列**候选处置**继续实现（未改 SDS 任何训练数值/权重/预算/门禁/PPO/课程/seed）；以下各项只涉及接口约定或 SDS 未逐字覆盖的语义。若裁决不同，改动点均为单点常量/字段，CPU 测试结构不变。
- 当前 CPU 证据（A 交付时点）：官方入口 `scripts/test_a2_ideal_stairs_v06.py --pattern 'test_v06_*.py' --maxfail 0` = **180 passed / 0 failed**（32.83 s，CPU-only）。

## Q001-1 `feet_air_time` 的腾空累计口径（**最重要，影响 SDS §3.4 数值锚点**）

- 问题：SDS §3.4 规则 2 定义 `filtered = raw OR previous_raw`；规则 4 说“脚已接触过地面，随后**无接触**时每步 `air_time += dt`”。SDS 未说明“无接触”指 `raw` 还是 `filtered`；但 §3.4 的数值验收写“腾空 `0.30 s` 后落地：单足奖励 `+0.0008`”。
- 现实现（A）：`air_time` 只在 `filtered == False` 时累计，落地支付用 `filtered` 上升沿，并额外保存 `previous_filtered_contact`（保证单帧失联不重复付款）。后果：物理上 15 个控制帧（0.30 s）无 raw 接触时，`filtered` 仍为真 1 帧，`air_time` 只到 **0.28 s**，实得 `0.2×(0.28−0.10)×0.02 = +0.00072`，**不等于 SDS 的 +0.0008**；要得到 +0.0008 需要 16 个 raw 腾空帧（0.32 s 物理）。
- 证据：`src/tasks/velocity/v06/rewards.py:314-410`（`AirTimeState.update`）；`tests/test_v06_rewards.py::ClearanceTests`/`test_exact_weighted_anchors` 目前用纯函数锁定 +0.0008/−0.0002，未锁定“经状态机的 0.30 s 物理腾空”。
- 候选处置（建议采纳 B）：`air_time` 按 **raw 无接触**累计（`air_time += dt` 当 `raw == False`），落地支付仍用 `filtered` 上升沿 + `previous_filtered_contact`。这样：15 帧 raw 腾空 → `air_time = 0.30`，落地（filtered 上升沿）支付 `+0.0008`；单帧 raw 失联 → filtered 保持真、无上升沿、不支付；站立不重复付款。
- 影响面：仅 `AirTimeState` 内部累计条件与两处测试；权重、阈值、支付公式、PPO、课程、门禁均不变。若采纳 A 现口径，则 SDS §3.4 的“0.30 s→+0.0008”需改写成“state-machine 0.30 s”。

## Q001-2 `RewardFrame` 增加每足地表高度 `foot_surface_height[B,4]`

- 问题：SDS §3.2 的 `h_i = z_site − 0.032 − H(x_foot,y_foot)` 需要每足脚下地表高度；W0 §4.2 冻结字段未列该字段。
- 现实现（A/B 已对齐）：追加为必需字段，由 `env._reward_frame()` 用同一 `TerrainSpec` 查询当前脚下地表（不平滑、不看下一阶）。
- 建议：批准（不删除/改名任何冻结字段，不改变奖励数值）。若要求严格逐字冻结，则改为 `surface_query` 回调，但这会引入 env→rewards 反向依赖。

## Q001-3 地形 tile 的世界布局与 `valid_bounds_local`（SDS 未给字面值）

- 问题：SDS §7.2 要求 `TerrainSpec.frame_origin_world/valid_bounds_local`，§5.2 未给世界布局/边界数值。
- 现实现（A）：沿用既有 v0.2 atlas 约定——局部 x=0 为第一级立面、y=0 为踏面中线、z=0 为地面；tile 世界原点 `(0, tile_id×8.0, 0)`；楼梯 bounds `(-2, N*d+2.0, -2, 2)`、平地 `(-2, 22, -4, 4)`。
- 建议：批准为几何约定（非训练数值；几何/地图/接触/奖励查询/成功与超时全读同一 `TerrainSpec`）。

## Q001-4 A2 默认关节角与软限位来源

- 问题：SDS §3.2 的 `q₀`、`joint_limits` 未打印数值。
- 现实现（A）：`q₀` 取 `src/assets/robots/unitree_a2/a2_constants.py::INIT_STATE`；软限位 = XML 硬限位 × 既有 `soft_joint_pos_limit_factor=0.9`；两者冻结进 `spec.REWARD_CONFIG`。
- 建议：批准“沿用现有 A2 资产配置”（影响 `similar_to_default`、`joint_limits` 与 `REWARD_CONFIG` SHA）。

## Q001-5 `--output-dir`/`--acceptance-file` 是否做字面一致性断言

- 问题：SDS §11.2 把二者列入“配方一致性断言”；但 `output_dir` 是运行目录。
- 现实现（A/C）：按 SDS 命令逐字冻结 `outputs=logs/a2_ideal_stairs_v06/train_seed42_trial01`、`acceptance_file=logs/a2_ideal_stairs_v06/implementation/V06_IMPLEMENTATION_ACCEPTANCE.md`，其他值拒绝；恢复走独立 `--resume-run`。
- 建议：批准字面断言（与“不换 seed/不加预算/不挑历史目录”一致）。若未来允许重跑新目录，需裁决后改 `CLI_DEFAULTS` 一处。

## Q001-6 超时公式回归锚点

- 问题：SDS §6.2 只给公式；实现需固定回归值。
- 现实现（A）：L40 `c=0.2→7257`、`c=0.6→2753`（h=d=0.20）作为公式守卫。
- 建议：批准为回归锚点（非新协议）。

## Q001-7 `diagnostic_consecutive` 的默认值（C 模块，需 Codex 确认）

- 问题：SDS §6.3 只规定“保持集连续两次 ≥12/15 失败 → `behavior_regression`”；未定义“诊断连续 N 次异常”常量。C 的 `speed.py` 对未定义项使用 fallback `2`。
- 建议：若沿用 r2 的 collapse 语义（连续 2 次），请明确写入；否则删除该 fallback，只保留 SDS 明文的保持集与最低子级规则。等 C 交付后补确切文件行号与证据。

## 边界

- 以上均不改变 SDS 的训练配方、预算、门禁、速度、课程、PPO 或 seed；未 commit；未启动 GPU。
- 依赖 Q001-1 的工作（`feet_air_time` 语义）会同时验证两种口径的可实现性，但在裁决前**不进入 GPU smoke/probe/训练**。
