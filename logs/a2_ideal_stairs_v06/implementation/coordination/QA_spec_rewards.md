# QA：v0.6 spec/rewards 问题与裁决落点（A 奖励子 agent）

- 生效合同 = SDS（`1ed4bf5f…6ad60`）+ Q001 裁决（SHA `5f3f2aa3…a4b1`）+ Q002 裁决（SHA `3ebc7441…cd52`）+ `W0_FREEZE_ADDENDUM_1.md`（当前冻结 SHA `b7553319…59ef`）+ `W0_FREEZE_ADDENDUM_2.md`（`c1dc2a8a…8dfc`），以 `W0_MANIFEST_v3.json` 为准。
- 本文件记录 SDS/W0 未逐字覆盖、但实现必须落地的接口或约定。A **未改动任何训练数值、权重、预算、门禁、PPO、课程或 seed**。
- Q001 裁决后各项状态如下；未裁决项仍等待 Codex。

## Q-A1（Q001-2，已裁决）`RewardFrame` 必需 `foot_surface_height`

- 问题：W0 §4.2 冻结字段没有每足地表高度，SDS 3.2 的 `H(x_foot,y_foot)` 无法计算。
- 处置：`RewardFrame` 增加必需字段 `foot_surface_height: float32[B,4]`；与 `foot_position_world` 同一世界 z 基准；由 env 用当前 tile 的同一 `TerrainSpec` 查询；缺失/non-finite/形状不符拒绝。未删除或改名任何冻结字段。
- 证据：`rewards.foot_clearance_rate`、`tests/test_v06_rewards.py::ClearanceTests`；B 的 `env.py::_reward_frame` 已提供该字段。

## Q-A2（Q001-3，已裁决）v0.6 自有世界布局，不再称“沿用 v0.2 atlas”

- 问题：SDS §7.2 要求 `frame_origin_world`/`valid_bounds_local`，但未打印世界布局数值。
- 处置：v0.6 自有平移布局：first tread edge local_x=0、平台起点 local_x=N*d、`frame_origin_world=(0, tile_id*8.0, 0)`；楼梯 bounds `(-2, N*step_depth+2, -2, 2)`、flat `(-2,22,-4,4)`。
- **措辞更正**：旧 v0.2 frame 的 x 是 `-10` 且 y 有整体偏移，因此只能称“一致平移”，不得称数值完全相同或“沿用旧 atlas”。`spec.py` docstring 与本节已按此改写。
- 证据：`spec.TERRAIN_TABLE`、`tests/test_v06_spec.py::test_q001_3_layout_wording_is_not_numerical_v02_equality`。

## Q-A3（Q001-4，已裁决）A2 q0 与软限位来自真实资产，软限位用中点公式

- q0：按 FL/FR/RL/RR × (hip, thigh, calf) = `(-.1,.9,-1.8, +.1,.9,-1.8, -.1,.9,-1.8, +.1,.9,-1.8)`，取既有 `a2_constants.py::INIT_STATE`。
- 软限位：**m=(lo+hi)/2，r=(hi−lo)/2，soft=(m−.9r, m+.9r)**（不是“硬限位端点×.9”）；`spec._soft_limits` 原实现正确，未改代码，只改注释措辞。数值：hip `[-.909,.909]`、前腿 thigh `[-2.0655,2.8755]`、后腿 thigh `[-1.285,3.665]`、calf `[-2.6585,-.6515]`。
- 运行时对照：`tests/test_v06_spec.py::RuntimeAssetTests` 构造真实 mjlab A2 Scene/Simulation（CPU），按关节名对照：
  - `default_joint_pos` max|delta| = **0.000e+00**
  - `soft_joint_pos_limits` max|delta| = **2.384e-07**（≤1e-6）
- 影响面：仅 `similar_to_default`/`joint_limits` 与 `REWARD_CONFIG` SHA。

## Q-A4（Q001-1，已裁决）air-time 按 raw 累计、filtered 边沿结算

- 处置：`AirTimeState.update` 逐字实现 Q001-1 伪代码：raw 失联首帧也累计；`landing` 需 raw 上升且 filtered 上升、`has_contacted_once` 且 `air_time>0`；raw 恢复无论是否付款都清零；落地帧不追加 dt；单帧失联无 filtered 上升沿故不付款。
- 状态字段：`previous_raw_contact / previous_filtered_contact / has_contacted_once / air_time` 全部保存/加载；缺字段、形状或 dtype 不符一律拒绝，不静默补零。
- 测试：`tests/test_v06_rewards.py::AirTimeTests` 经状态机逐帧覆盖 8 类场景（15×False→+0.0008 且不重复、True→False→True=0、多次单帧失联、出生首触、3×False→−0.00016、四足异步、部分 reset、两帧失联、5 N 严格阈值、state_dict 中途恢复/拒绝缺字段）。0.05 s→−0.0002 仅作为 dt=.02 的纯函数锚点，不称 2.5 个状态机帧。
- 影响面：仅 `feet_air_time` 状态内部；权重、阈值、公式不变。

## Q-A5（Q001-5，已裁决）固定 fresh 输出与验收路径

- `output_dir=logs/a2_ideal_stairs_v06/train_seed42_trial01`、`acceptance_file=logs/a2_ideal_stairs_v06/implementation/V06_IMPLEMENTATION_ACCEPTANCE.md`；允许解析后等价绝对路径，不允许其他 run/报告；目录已存在拒绝且不覆盖；恢复只走 `--resume-run`；不授予 trial02。
- 落点：`spec.CLI_DEFAULTS`/`recipe_config` 字面断言；`tests/test_v06_spec.py::RecipeCliTests`。

## Q-A6（Q001-6，已裁决）超时公式锚点

- `episode_timeout_steps(40,.20,.20,.2)=7257`、`(...,.6)=2753`，作为公式守卫（非查表替代通式）；D25 参考例 1874。
- 证据：`tests/test_v06_spec.py::CurriculumTests::test_timeout_formula`。

## Q-A7（Q001-7，已裁决）`DIAGNOSTIC_COLLAPSE_STREAK=2`

- `spec.DIAGNOSTIC_COLLAPSE_STREAK=2` 显式定义并纳入 `REWARD_CONFIG['diagnostics']` 与 contract，随之进入 `reward_config_sha256`。
- 作用域仅“继承的停滞 collapse”，不扩展到任意诊断异常；flat/H00 保持集与最低子级 4 窗口计数独立保存。诊断的 15 例、3 档各 5、两项≥12、当前 generation≥100 更新等语义由 C 的 speed/runner 消费；缺常量时预检报错，无静默 fallback。
- 证据：`tests/test_v06_spec.py::RewardTableTests::test_diagnostic_collapse_streak_in_reward_config_and_sha`。

## Q-A8（Q002-2，已裁决）`FALL_DROP = 0.50` 与期望踏面判据

- `spec.FALL_DROP = 0.50`（米）为必需常量；`events.py`/`env.py` 直接读取，不得静默 `getattr` 默认。
- 文档化判据（当前 tile frame）：`region=(N>0)&(local_x>=0)`；`z_expected=min(max(floor(local_x/d)+1,0),N)*h`；`below_support=region&(local_z<z_expected-0.50)`（严格小于）；平台 `z_expected=N*h`；flat 与 `local_x<0` 不启用；不得改为足端接触面/四足平均支撑/出生高度。
- 入 SHA：`REWARD_CONFIG['fall_drop']`、`contract['fall_drop']` 均含 `drop=0.50`，随 `reward_config_sha256`/contract SHA 冻结。
- 测试：`tests/test_v06_spec.py::RewardTableTests::test_q002_2_fall_drop_constant_formula_and_sha`（常量、strict、flat/approach 关闭、平台期望 N*h、改值改变 SHA）。

## Q-A9（Q002-4，已裁决）八类停止状态 + 三生命周期状态

- `STOP_STATES`（顺序冻结）：`behavior_regression`、`behavior_collapse`、`curriculum_stalled`、`budget_exhausted`、`gate_failed`、`user_requested_stop`、`monitoring_blocked`、`resource_insufficient`。
- `LIFECYCLE_STATES`：`running`、`ready_for_final_test`、`complete`；`complete` 仅表示最终三速×100 例通过；`pilot_not_applicable` 不是 v06 实际终态。
- 已入 `contract['states']`（stop/lifecycle/all/complete_meaning/pilot_not_applicable_is_terminal），contract SHA 覆盖；C 的 runner/recovery/orchestration 引用同一常量。
- 测试：`tests/test_v06_spec.py::RewardTableTests::test_q002_4_stop_and_lifecycle_states`。

## Q001-7 确认

`DIAGNOSTIC_COLLAPSE_STREAK=2` 同时存在于 `REWARD_CONFIG['diagnostics']`、`contract['diagnostics']`，并已覆盖于 `reward_config_sha256`；测试见 `test_diagnostic_collapse_streak_in_reward_config_and_sha`。

## Q-A10（Q001-7 / SDS 7.1）继承 r2 停滞诊断常量移入 spec

- 为满足 SDS 7.1“spec 是唯一数值源”，`spec.py` 显式定义 B 的 `env.py` 原先硬编码的六个继承常量（值逐字保持 r2 原值，不改数值/行为）：
  `STALL_HISTORY_SAMPLES=51`、`STALL_DX_FRACTION_PER_SECOND=.15`、`STALL_DX_THRESHOLD_SCALE=1.0`、`STALL_HIGHWATER_DX=.03`、`STALL_CONSECUTIVE_SAMPLES=50`、`STALL_ACTIVE_COMMAND=.1`。
- 语义：继承 r2 的停滞诊断定义，**仅诊断、不付款、不终止**（Q001-7 关闭旧 stall 付款）；B 的 env 改为从 spec 读取，删除硬编码。
- 入 SHA：`REWARD_CONFIG['diagnostics']['stall']` 与 `contract['diagnostics']['stall']`（含 `reward_payment=False`、`termination=False`），随 `reward_config_sha256`/contract SHA 冻结。
- 测试：`tests/test_v06_spec.py::RewardTableTests::test_inherited_stall_diagnostic_constants_in_sha`（六个值、config/contract 一致性、改值改变 SHA）。
