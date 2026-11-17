# QB：B（地形/环境/课程）实现说明、依赖与待裁决事项

- 提交人：实现子 agent B；日期：2026-11-16。
- 只写本文件；未修改 `spec.py`、`rewards.py`、SDS、审计、审核、R1/R2 日志与 `control.json`。
- 本文件不代表验收通过。

## 1. 已完成并自测的接口（与 W0 §4.3/§4.4 对齐）

| 模块 | 接口 | 说明 |
|---|---|---|
| `v06/terrain.py` | `build_terrain_specs()`、`build_terrain(seed)`、`tile_geoms()`、`query_surface()`、`query_surface_height()`、`platform_bounds()`、`index_geometry()`、`timeout_steps()`、`terrain_generator_cfg()` | 22 tile 精确盒体，世界坐标由 `spec.TERRAIN_TABLE.frame_origin_world` 绝对定位；单 patch 生成器（`num_rows=1,num_cols=1`），不依赖 v02 网格/ATLAS |
| `v06/map.py` | `HeightMap`、`ProprioHistory` | 3×48×32、10 Hz；高度只来自 v06 terrain 查询 |
| `v06/contacts.py` | `ContactState` | `raw=found AND f_z>5N`、`filtered=raw|prev`；接触归约在 v06 内自带实现（不再 import `v02.contacts`） |
| `v06/events.py` | `evaluate()`、`climb_success()`、`failure_masks()`、`timeout_steps()`、`first_step_contact()`、`euler_heading()` | 成功/失败/超时分离；8 个失败 mask；`N*当前踏深` |
| `v06/env_cfg.py` | `env_cfg(num_envs, seed)`、`v06_env_cfg` | 地形/接触/传感器/观测全部走 v06 模块 |
| `v06/env.py` | `V06Env`、`ScheduleAssignment`（从 curriculum 再导出）、`NumericalFailure` | 冻结签名、按更新边界 `set_schedule` 部分 reset、逐 env outcome、奖励只调用 A 的 `RewardComputer` |
| `v06/curriculum.py` | `CurriculumState`、`EpisodeOutcome`、`TransitionEvent`、`CurriculumManager`、`ScheduleAssignment`、`allocate_roles()`、`role_tiles()`、`retention_gate_kind()` | 64 FIFO、EMA、generation、H tick、D/L 子级预算、V/R 门槛；保持集按 Q001-7 独立 |

## 2. spec.py 依赖（A 已提供，实际绑定如下）

- 地形：`TERRAIN_TABLE`、`TerrainSpec`、`terrain_spec`、`tile_id_for`、`FLAT_TILE_ID`、`STAIR_WIDTH`、`TREAD_THICKNESS`、`PLATFORM_LENGTH`、`STAIR_QUERY_*`。
- 传感器/接触：`FOOT_ORDER`、`SUPPORT_FORCE_GT`、`SUPPORT_DEBOUNCE`、`NONFOOT_PERSIST_TIME`、`ATTITUDE_LIMIT_RADIANS`、`SAFE_INSET`、`FOOT_RADIUS`、`F0_EPISODE_STEPS`。
- 成功/超时：`SUCCESS_*`、`PLATFORM_LENGTH`、`STAIR_TIMEOUT_BASE_M`、`STAIR_TIMEOUT_SECONDS`。
- 课程：`CURRICULUM_*`、`ADVANCE_*`、`REGRESS_*`、`COOLDOWN_CHECKS`、`ROLE_*`、`NEAR_OFFSETS`、`EASY_BACKOFF`、`V_*`、`R_*`、`SUBSTAGE_*`、`MIN_LEVEL_*`、`RETENTION_*`、`COMMAND_*`、`SPAWN_*`、`PHASES`/`PHASE_ORDER`。
- 下坠阈值：必需常量 `spec.FALL_DROP = 0.50`（A 已落地；`events.py` 直接读取、无静默默认）。

## 3. 关键实现选择（需 C/主对话接线时知情）

1. **自闭合基类**：为满足“v06 不得运行时 import v02/v02_formal”（主对话 2026-11-16 集成反馈），`V06Env` 直接继承 `mjlab.envs.ManagerBasedRlEnv`，在 v06 内实现与 `v02_formal.env.FormalEnv` 相同的生命周期（构造参数、reset/step 形状、reset 前 terminal 快照、部分 reset）。W0 §4.3 的 `V06Env(FrozenEnv)` 语义保留，但不再是字面继承。
2. **地形世界布局**：每个 tile 的碰撞盒位置 = `frame_origin_world + 局部坐标`；生成器只做 1×1 patch（`world_position=(-12,-4)`），因此 tile 世界布局完全由 spec 决定，不会隐式跟随 v02 网格。flat tile 的 `width` 在 spec 中是 8.0（tile 间距），几何校验只对 `N>0` 强制 0.80。
3. **`query_surface_height`**：按该位置所有盒体的最高顶面解析求值（含 `|y|<=width/2` 的踏板足迹），与 `mj_ray` 在 22 tile 的 tread/platform/approach 上逐点一致（容差 2e-6）；越界返回 NaN，另有 `query_surface()` 返回 validity。
4. **平台安全区**：`platform_bounds(tile)=(N*d+0.042, N*d+2-0.042, -0.358, 0.358)`；flat 无平台返回 4×NaN。
5. **成功判据**：冻结 R2 平台接触定义，`N*0.20` 全部替换为 `N*当前踏深`；单足 2 帧去抖后再计 5 连续帧（因此从首次平台接触算起需 6 个控制帧）。失败优先级在 `events.evaluate()` 内实现（`success &= ~failure`），奖励事件由 A 的 `event_rewards` 再次保证失败优先。
6. **奖励接线**：`env._reward_frame()` 提供 W0 §4.2 全部字段，并额外提供 A 的 Q001-2 字段 **`foot_surface_height[B,4]`**（v06 精确查询 `H(x_foot,y_foot)`）；环境只调用 `v06.rewards.RewardComputer(frame)`，**未复制任何奖励公式**（raw/weighted/total 全部来自 A）。
7. **方向/角度**：`euler_heading`、`wrap_to_pi` 在 `v06/events.py` 内实现，与冻结约定一致（pitch 取反）。
8. **`set_schedule`**：只在更新边界调用；只 reset `(phase, substage, role, tile)` 变化的 env；受影响的接触、地图、历史、奖励差分、腾空状态全部按 env 局部 reset（`RewardComputer.reset(ids)`、`HeightMap.reset(ids)`、`ContactState.reset(ids)`、`PlatformSupportState.reset(ids)`）。
9. **角色/地形抽样**：角色分配确定性最大余数（平手 flat→frontier→near→easy）；每 episode reset 时 near/easy 用独立 `numpy` 抽样器（seed+7919）重抽，frontier/flat 固定；退级后按新 k 截断（`role_tiles()` 内）。抽样逻辑在 `curriculum.role_tiles()` 单一实现，env 复用。
10. **FIFO/EMA/generation**：FIFO 按 `(end_control_step, env_id)` 排序（env 记录新增 `end_control_step`，`EpisodeOutcome` 优先读它，其次 `terminal_step`）；每次 `on_update` 最多消费一个 64 窗口；首批 EMA=batch，之后 0.97/0.03；任一行/子级/阶段变化即新 generation，清 EMA/FIFO/ready/hard/gate streak，cooldown=5，旧 generation 结果不入 window。
11. **gate 提交时序**：`record_gate()` 只登记（含 generation/phase 校验），实际 phase/substage/level 转换在**下一次** `on_update()` 提交，保证真实地形只在完整更新边界变化。C 的边界顺序（`on_update` 在 `run_gate` 之前）与此兼容。
12. **RNG 接口**：`env.rng_state()/load_rng_state()`（spawn/command torch generator）与 `env.sampler_rng_state()/load_sampler_rng_state()`（terrain 抽样 numpy bit generator），与 C 的 `recovery.RNG_FIELDS=('env','sampler')` 对应；已用 C 的 `recovery.capture_rng/validate_rng/restore_rng` 实测通过。
13. **`ScheduleAssignment`** 增加 `generation` 字段（W0 冻结字段全保留），用于 episode 记录打 generation 标签。

## 4. 待 Codex 裁决 / SDS 未覆盖

### 4.1 F0→H 的晋级条件（Q002-1 已裁决，已实现）
F0 同 generation 至少完成 100 更新后，连续两次完整门禁通过（速度 .3/.5/.8，每档 20 例、每档 ≥18）才进入 H；最早 F0 更新 200 可晋级，不是额外固定配额。有效失败清 streak，评估异常不得当通过。

### 4.2 下坠判据（Q002-2 已裁决，已实现）
`FALL_DROP` 是 spec 必需常量（0.50 m），`events.py` 直接 `float(spec.FALL_DROP)`，无 getattr 默认。判据为冻结 v0.2 的**期望踏面**（不是最近支撑面/四足平均面/出生高度），以当前 tile frame 的 local_z：

```
region = (N > 0) & (local_x >= 0)
z_expected = min(max(floor(local_x / d) + 1, 0), N) * h
fall = region & (local_z < z_expected - spec.FALL_DROP)   # 严格 <
```

平台区 `z_expected=N*h`；flat 与 `local_x<0` 不启用；相等不触发。测试覆盖跨阶边界、平台、flat、local/world 平移、等号与 `torch.nextafter` 两侧，以及下坠 failure 进入课程 F。

### 4.3 保持集接口与调度（Q001-7 + Q002-6 已裁决，已实现）
两个保持集完全独立，调度字段（JSON-safe，均在 `state_dict()` 内）：

| 字段 | 类型 | 语义 |
|---|---|---|
| `retention_streaks` | `{'flat': int, 'h00': int}` | 各自连续失败计数 |
| `retention_enabled` | `{'flat': bool, 'h00': bool}` | 是否已取得资格 |
| `retention_qualified_at` | `{'flat': int\|None, 'h00': int\|None}` | 首次取得资格的 global update q |
| `retention_next_due` | `{'flat': int\|None, 'h00': int\|None}` | 严格下一个 200 倍数：`200*(floor(q/200)+1)`，每完成一次 `+= 200` |
| `retention_last_completed` | `{'flat': int\|None, 'h00': int\|None}` | 最近完成的检查 update（同一 update 幂等，恢复不重复累计） |

- `retention_due(global_update) -> tuple[str, ...] | None`：返回本边界**全部**到期短名，固定顺序 `('flat','h00')`；无到期 `None`；刚取得资格的同一 boundary 不返回；退级/generation 不清、不推迟调度字段。
- `record_retention(kind=..., failed_cases=..., passed=None, update=None)`：kind 接受短名或 gate 名；`update` 缺省取 `state.last_update`；只更新对应集合；完成时 `next_due` 前进到严格下一 200 倍数并写 `last_completed`；同一 `(kind, update)` 重复调用返回 `duplicate=True` 且不重复计数；返回 `dict(kind, gate_kind, failed_cases, cases, streak, required, stop, qualified_at, next_due, last_completed, duplicate)`。`stop` 仅表示该集合自身连续 2 次 ≥12/15，`behavior_regression` 停机由 C 的 runner 执行。
- `RETENTION_GATE_KINDS = {'flat': 'flat_retention', 'h00': 'stair_retention'}`、`retention_gate_kind(kind)` 供 C 的 `run_gate(kind)` 映射；`str(due)` 单值写法必须改。
- 已测：q=200→400、q=350→400、q=400→600；退级/generation 不推迟；恢复补完不重复、不跳过；两集合同边界都执行。

### 4.4 C 的 runner 尚未消费课程终止状态（集成风险，待 C/主对话处理）
`CurriculumManager.state_dict()['finished']/['stop_status']` 会给出 `complete`（R 两次门禁）、`minimum_level_failure`（最低子级 4 窗口 F_EMA≥0.80）与 `budget_exhausted`。C 当前 `_check_stop_conditions()` 不读这两个字段，也可能在 R 通过后继续训练到总预算。B 侧已发出 `TransitionEvent(kind='final', status=...)`，请 C 接线或由主对话裁决。

### 4.5 H00/H15 两次门禁的含义（已按字面实现，供核对）
- H00：`level_tick==0` 时连续 2 次 counted gate 通过 → 解锁自适应升级；同一次通过会置 `retention_enabled['h00']`。
- H15：`level_tick==100` 且连续 2 次 counted gate 通过 → 进入 D（D25），gate 必须来自同一 generation 且该 generation 已 ≥100 更新。
- 任何 generation 切换都清 gate streak；回归到 tick 0 后须重新取得两次门禁。

### 4.6 成功判据的 5 帧计数起点（已按冻结实现，供验收复算）
单足平台支撑需连续 2 帧（首帧只建状态），此后“连续满足 5 控制帧”才成功，因此从首次平台受力算起是第 6 个控制帧。若验收认为第 5 帧即应成功，需要改冻结判定语义（B 不擅自改）。

## 4.7 停滞诊断（Q001-7 / SDS 6.3，已实现；只诊断不付款）

- `v06/env.py` 内自带 `StallDiagnostics`（r2 语义移植，未 import v05_r2/v02），常量在 `env.py` 顶部注明继承自 `v05_r2.spec`：`STALL_HISTORY_SAMPLES=51`、`STALL_DX_FRACTION_PER_SECOND=0.15`、`STALL_DX_THRESHOLD_SCALE=1.0`、`STALL_HIGHWATER_DX=0.03`、`STALL_CONSECUTIVE_SAMPLES=50`、`STALL_ACTIVE_COMMAND=0.1`。**v06 spec 目前没有这些常量**，若要求“所有数值只来自 spec”，需 A 补常量；当前按 Q001-7“沿用 r2 定义”实现，未改任何文件。
- 判定：`active = enabled & c>0.1 & ~climbed`；`qualifying = active & first_step_acquired & window_ready & dx<threshold & dm<highwater`（父任务要求“首阶已获后仍未推进”；r2 原式为 `~climbed`，此处按父任务加首阶门控）；`penalty_on = qualifying & consecutive>=50`。`stall_samples += penalty_on`、`active_samples += active`；每步最多一次、按 env 独立；部分 reset 只 reset 对应 env（`stall.reset(ids, local_x0)`）。
- 不付款：env 的奖励帧不包含 stall 字段，`step_reward_tensors` 无 `stall*` 键，`reward == Σweighted` 不受 stall 状态影响。
- 输出字段（C `evaluation.py:506-525` 读取）：
  - episode 记录：`active_samples`（int）、`stall_samples`（int ≤ active）、`first_step_valid`（bool，同时保留 `first_step_acquired`）、`control_samples`、`max_abs_y`、`max_abs_yaw_deg`、`forward_distance`、`flat_pass`；
  - `info`（无 episode 的 F0/诊断 case 走这里）：同名字段均为**标量**（env 0，case 复制到 8 env 时等价），`info['diagnostics']` 同名字段也是标量，避免 C 的 `int()`/`float()` 失败；
  - `env.diagnostics`：上述标量 + `*_by_env` 张量（`active_samples_by_env`/`stall_samples_by_env`/`first_step_valid_by_env`）+ 现有张量、`success`/`failure`/`timeout`、`stall_penalty_on`、`stall_dx_1s`；
  - `env.diagnostics_cases()`：逐 case（逐 env）列表，含 `env_id/role/tile_id/vx/step/active_samples/stall_samples/first_step_valid/stall_penalty_on/max_abs_y/max_abs_yaw_deg/forward_distance`（JSON-safe）。
- `forward_distance = max_body_x - episode_initial_local_x`（本回合最大前进位移；未验证 SDS 是否要求净位移，若 Codex 另有定义按裁决改）。
- 测试：原地停滞 `stall_samples` 增长且 `active` 同步增长；正常前进 `stall=0`；未获首阶不判停滞、获首阶后停滞；env 级 110 步停滞计数与“无 stall 奖励项、reward=Σweighted”；F0 case 500 步输出 `flat_pass/max_abs_y/max_abs_yaw_deg/forward_distance` 标量。

## 4.8 评估前沿地形（SDS 6.1，已实现）

- `V06Env.__init__` 增加 `level_tick: int = 0`、`frontier_tile_id: int|None = None`：
  - 显式 `frontier_tile_id`：所有 env 固定该 tile（roles 仍按 phase 分配），`_sample_tiles` 不再随机抽样，reset 也保持单地形；
  - 否则 H 用 `frontier_tile_for_state(phase, substage, level_tick)`；`evaluation=True` 时即使未显式给 tile，也把解析出的 frontier 固定为单地形（不再混合 near/easy）；
  - `evaluation=True & phase='H' & frontier_tile_id is None & level_tick==0` 直接 `ValueError`（禁止静默用 H00 评估）；
  - 显式 tile 会校验 phase/substage 合法（H 允许 tile 1..16；D/L 必须对应子级；V/R 必须 21）。
- `set_schedule` 的 frontier 校验原为“必须等于 level_tick=0 的 tile”，会阻塞 H tick≥7 的训练；现改为 `_check_schedule_frontier`（按 phase/substage 校验，H 接受当前 tick 的任何合法 T0..T15），训练路径仍由课程驱动。
- 逐 case/诊断/episode 记录新增 `frontier_tile_id`（`tile_id` 原有）；`info`、`env.diagnostics`、`diagnostics_cases()`、`diagnostics_snapshot()` 均带该字段。
- C 需同步：`case['tile_id']` 传入 `build_case_runner(frontier_tile_id=...)`；未验证 C 侧实际接线。

## 4.9 F0 前进距离（Q003 已裁决，已实现）

- `forward_distance` 统一为**有符号净位移** `D_net = x_local[t] − x_start`，x 为 case 所属 tile 坐标系机身前向；起点是出生写入 + `sim.forward/sense` 后、首次动作前的**实际**位置（`episode_start_local_x`，不是常数 −1/-0.9），不 clamp、不 abs、不用积分速度。
- 每控制步批量计算 `forward_distance_buf = state.local[:,0] − episode_start_local_x`；终止时（success/failure/timeout）在 `_reset_ids` 之前写入**不可变快照** `frozen_forward_distance/start/end/steps`。自动 reset 不清理该快照，新 episode 的 0 起点不会覆盖已冻结 case 结果。
- 输出：`info` 与 `diagnostics` 均带 per-env 张量 `forward_distance`、`forward_distance_start`、`forward_distance_end`、`forward_distance_terminal`、`forward_distance_terminal_start/end`、`max_forward_excursion`、`flat_pass`（**不再用 env0 标量冒充**）；逐 case 标量在 `diagnostics_cases()`/`diagnostics_snapshot()`/episode 记录。
- `max_body_x` 保留原统计含义（历史最大 local x）；最大前进位移另名为 `max_forward_excursion`，只展示、不用于门禁；累计正向路程未实现。
- `env._flat_pass` / 纯函数 `flat_case_verdict()`： `steps==500 & ~failure & max|y|<=.20 & max|yaw|<=10° & D_net >= .5*c*10 − 1e-9`；与 C 的 `speed.flat_case_pass` 逐例一致（测试直接对拍）。
- 测试（test_v06_env.py）：Q003 §3 六类净位移（1.5 通过 / peak1.0 net0 不通过 / 往返累计>1.5 net0 不通过 / 先退后进 1.6 通过 / 净后退 −0.2 不通过且字段负 / end 1.499999 不通过）；起点 −0.9/−1.1 同净同判；世界平移不变；c=.5/.8 门槛 2.5/4.0 恰好/差 1e−6；failure/y/yaw/499 步否决；500 步 env 级 `flat_pass` 为真；自动 reset 前冻结旧起终点且 batch 另一例不受污染；499 步提前失败不拼接下一 episode；x0/x500 trace 复算一致。

## 4.10 证据出口的独立终点（Q005 §3.4 已裁决，已实现）

- 所有非终止证据出口（`info`、`info['diagnostics']`、`env.diagnostics`、`diagnostics_cases()`、`diagnostics_snapshot()`、episode 记录）的 `forward_distance_end` 现在直接取**同一帧真实 `state.local[:,0]`**（`forward_distance_end_buf`），`forward_distance_start` 取本 episode 出生后首帧真实 local x，`forward_distance = end − start`（保留符号）。不再用 `start + D` 回算 end。
- 终止/自动 reset：先由同一帧的 `state.local[:,0]` 冻结 `frozen_forward_end` 与 `frozen_forward_start`，再由两者相减得到 `frozen_forward_distance`，最后才 `_reset_ids`；快照跨 reset 保留。
- 奖励/观测/物理/终止/课程判定未改；`max_body_x` 与 `max_forward_excursion` 语义不变。
- 测试（test_v06_env.py）：F0 500 步逐帧记录真实 local x，末帧与 `info['forward_distance_end']`/diagnostics 一致；500 步末帧与 `distance=end−start` float32 一致；8 环境各自 start/end 互不相同且与逐 env 观测一致；净后退 end<start 且 `forward_distance<0`；terminal/自动 reset 的 episode 记录 end 来自 reset 前真实值。

## 5. 未验证事项

- 未启动任何 GPU / 训练 / 仿真；仅 CPU 单元测试与 2–4 环境 CPU 场景构造。
- 未跑 C 的 runner/PPO 集成（只验证了 `recovery.validate_curriculum/capture_rng/restore_rng` 与 `env.set_schedule/step` 的接口）。
- 未跑 GPU smoke/probe；未验证 512 环境性能（10 Hz 地图与 22 tile 查询的 GPU 时延）。
- `contact overflow` 当前作为失败 mask 锁存并跳过该步接触采样；未在真实 overflow 场景验证后续行为（SDS 未给“溢出后如何继续”）。
- 保持集新接口与新调度字段尚未在 C 的 runner 实测（C 侧仍需按 §4.3 改为消费列表并传 kind）；B 侧只做 CPU 单元测试。
- V ramp 的 `u=阶段内已完成更新数`（Q002-7）由 C 传入的 `stage_iteration`/`set_schedule` 决定；B 只按 `stage_iteration` 计算命令范围，B 侧未验证 C 的 resume 计数口径。
- 评估前沿：B 的 env 已支持 `level_tick`/`frontier_tile_id` 单地形评估，但 C 的 `case['tile_id']` 与 `build_case_runner(frontier_tile_id=...)` 尚未接线，未做端到端验证。
- 全量 `scripts/test_a2_ideal_stairs_v06.py` 未运行（A/C 文件仍在并行修改）。

## 6. 复现命令

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/xxs/anaconda3/envs/a2/bin/python -m pytest \
  tests/test_v06_terrain.py tests/test_v06_contacts_events.py \
  tests/test_v06_env.py tests/test_v06_curriculum.py -q
```
