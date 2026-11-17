# A2 楼梯训练 v0.6 SDS：参考 Go2 项目的奖励与渐进课程

**目标：** 用两个参考项目中的速度跟踪、楼梯姿态约束、地形相对抬脚奖励和难度混合训练，替换原来偏重抬脚、落阶事件的方案。先学会持续上楼，再加入低速指令，最终完成原定的 **40 级、级高 0.20 m、踏深 0.20 m、宽 0.80 m 开放楼梯**。

**执行方式：** DSH 主对话协调，执行子 agent 分工修改，另设独立验收子 agent。数值、课程或验收遇到未覆盖问题时，向本 Codex 提问，不自行调整。

**本文件替代此前未实施的 0.6 草案。** 不采用此前草案中的双优化器方案。

**当前状态：** 用户已明确批准执行（Implement the plan）。本 SDS 为批准的执行规格，允许 DSH 按 W0→工程检查→独立验收→新 v0.6 seed42 连续训练实施。此授权不代表工程验收或训练效果已经通过。

**修订：** 1.0；批准日期：2026-11-16。数值与流程保持已批准方案；本次落盘仅更新执行状态。

---

## 1. 参考依据与采用范围

固定参考版本，实施时不得跟随上游 `main` 自动更新。

| 参考项目 | 固定 commit | 本版采用内容 |
|---|---|---|
| `ashutoshpanpalia/unitree_go2_stairs` | `85d4145e84cd7dd2d8668c7804db92b381fde14d` | 高斯速度跟踪、连续运动约束、0.30 m 踏面作为初期楼梯 |
| `saifahmadgit/go2-sim2real-locomotion-rl` | `0830f3e3582acd5c9b3a5c09186d4f028f54d9dd` | 楼梯奖励的主要权重、只惩罚侧倾、竖直速度死区、地形相对高度、从行走策略初始化、缓升快退、当前/临近/简单地形混合 |

主要源码：

- [第一个项目的奖励实现](https://github.com/ashutoshpanpalia/unitree_go2_stairs/blob/85d4145e84cd7dd2d8668c7804db92b381fde14d/source/unitree_go2_stairs/unitree_go2_stairs/tasks/stairs_walking/unitree_go2_stairs/unitree_go2_stairs_env.py#L206)。
- [第二个项目的楼梯训练配置](https://github.com/saifahmadgit/go2-sim2real-locomotion-rl/blob/0830f3e3582acd5c9b3a5c09186d4f028f54d9dd/examples/locomotion/final/go2_train_stair.py)。
- [第二个项目的奖励、课程与地形采样实现](https://github.com/saifahmadgit/go2-sim2real-locomotion-rl/blob/0830f3e3582acd5c9b3a5c09186d4f028f54d9dd/examples/locomotion/final/go2_env_stair.py)。

必须在交付文档中区分“上游原值”和“A2 适配值”：

1. 第二个项目当前源码抬脚目标为 `0.17 m`，发布模型旁的保存配置为 `0.12 m`。本版参考当前源码，不能宣称 `0.17 m` 已由发布模型验证。
2. 第二个项目的腾空奖励存在先清零、后结算的问题，本版修正结算顺序。
3. 第二个项目的前进奖励返回位移后又乘 `dt`。本版明确采用每米 `+0.4`，不复制重复缩放。
4. 第二个项目用超时生存率推进课程。本版改为真实安全登顶率，不能把站着活到超时算作成功。
5. 第一项目的当前任务代码没有完整的逐级晋退实现，不能把 README 的课程描述当成已实现算法。
6. 不复制 Go2 的机身尺寸、16 动作可变刚度、扭矩上限或仿真器。A2 保持现有 12 关节动作与动力学。

---

## 2. 版本、初始化与不变条件

### 2.1 版本身份

| 字段 | 值 |
|---|---|
| `code_version` | `0.6.0` |
| `recipe_version` | `0.6-go2refs-r1` |
| `schema_version` | `8` |
| `architecture` | `v06-route-state-v1` |
| `task` | `Unitree-A2-IdealHeightMap-Stairs-V06` |
| `protocol` | `v06-go2refs-climb-first-v1` |
| 日志根目录 | `logs/a2_ideal_stairs_v06/` |
| 训练模式 | `train`，单条连续训练 |
| 训练 seed | `42` |

新增文档：

- `doc/sds/SDS_A2_IdealHeightMap_Stairs40_v0.6.md`
- `doc/sds/PROMPT_A2_Stairs_v0.6_DSH_Execution.md`
- `doc/sds/reviews/SDS_v0.6_reference_audit.md`
- `doc/sds/reviews/SDS_v0.6_independent_review.md`

### 2.2 初始化

从以下 ETH 模型重新初始化，不接着 R2 失败模型训练：

```text
/home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt
SHA256:
8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722
```

迁移范围保持：

- 迁移 ETH Actor MLP 和冻结的 45 维归一化器。
- 高程图、历史与路线融合旁路初始为零，确保初始动作均值与 ETH 对应。
- Critic 重新初始化。
- Adam、课程计数、训练计数重新初始化。
- 动作标准差按本 SDS 设为 `0.30`，不继承 ETH 的标准差。

### 2.3 保持不变

- A2 12 维关节位置动作，动作比例 `0.25`。
- 控制步长 `0.02 s`；物理步长 `0.005 s`；每控制步 4 个物理子步。
- Actor 输入：`proprio[45]`、`history[5,45]`、`heightmap[3,48,32]`、`route_error[3]`。
- Critic 输入宽度 `211`；privileged 状态宽度 `35`。
- 路线观测：
  `route_error=[clip(local_y/0.20,-4,4), sin(e_world), cos(e_world)]`。
- 高程图更新频率 `10 Hz`。
- 现有 A2 PD、关节限位、碰撞模型和接触判定安全规则。
- 最终楼梯的开放立面、开放侧边、踏板厚 `0.01 m`、顶部平台长 `2.0 m`。
- 不恢复 R2 seed43，不修改其停止文件和历史证据。
- 不增加摩擦、质量、PD、推力或动作延迟随机化。本版先处理上楼任务；R 阶段只扩大初始位置和航向扰动。

---

## 3. 奖励函数：完整公式与数值

### 3.1 总奖励

每个控制步：

\[
r_t=dt\sum_i w_i f_i
+0.4(x_t-x_{t-1})
+10I_{\text{success}}
-10I_{\text{failure}}
\]

其中：

- `dt=0.02`，连续项只乘一次。
- `x` 为所属楼梯 tile 坐标系的前进位置。
- 成功、失败是一次性终端事件，不乘 `dt`。
- 失败优先：同帧成功和失败条件同时出现，只给失败奖励。
- 超时额外奖励为 `0`。
- 总奖励不做正值截断。
- 所有奖励使用本物理帧实际执行的指令及 reset 前状态。

### 3.2 连续奖励表

记：

- `v、ω`：机体系线速度、角速度。
- `c`：实际前进指令；横移和转弯指令均为零。
- `g`：机体系归一化投影重力。
- `q₀`：现有 A2 默认关节角。
- `H(x,y)`：当前 tile 在该位置的最高地表高度。
- `Cᵢ`：第 i 足的滤波接触状态。
- `hᵢ=z_site,i−0.032−H(x_foot,i,y_foot,i)`：足底离地净空。
- `uᵢ`：第 i 足世界系水平速度。

**下表数值全部是乘 `dt` 前的权重。**

| 名称 | 权重 | 未加权公式 |
|---|---:|---|
| `tracking_lin_vel` | `+1.5` | `exp(-((v_x-c)²+v_y²)/0.25)` |
| `tracking_ang_vel` | `+0.8` | `exp(-ω_z²/0.25)` |
| `lin_vel_z` | `−1.0` | `max(abs(v_z)-0.15,0)²` |
| `orientation_roll` | `−5.0` | `g_y²` |
| `base_height` | `−0.1` | `(z_base-H(base_xy)-0.40)²` |
| `action_rate` | `−0.01` | `Σ(a_t-a_(t-1))²` |
| `similar_to_default` | `−0.05` | `Σabs(q-q₀)` |
| `dof_acc` | `−2.5e−7` | `Σ((qdot_t-qdot_(t-1))/dt)²` |
| `dof_vel` | `−5e−4` | `Σqdot²` |
| `ang_vel_xy` | `−0.05` | `ω_x²+ω_y²` |
| `feet_air_time` | `+0.2` | `ΣI_first_contact,i × (air_time_before_reset,i-0.10)` |
| `foot_slip` | `−0.15` | `ΣCᵢ × ‖uᵢ‖²` |
| `foot_clearance` | `−0.5` | `Σ(1-Cᵢ) × (hᵢ-h_target)² × ‖uᵢ‖` |
| `joint_tracking` | `−0.1` | `Σ(q_target_executed-q)²` |
| `center` | `−0.5` | `min((local_y/0.20)²,4)` |
| `heading` | `−0.5` | `1-cos(e_world)` |
| `joint_limits` | `−10.0` | `Σ[max(q_min_soft-q,0)+max(q-q_max_soft,0)]` |

这些项中，速度、姿态、步态与平滑项主要来自第二个项目；`center`、`heading`、`joint_limits` 保留 A2 当前任务约束。

### 3.3 高度目标

- 机身高度目标固定为 **`0.40 m`**，来自本地 A2 初始配置，不使用 Go2 的 `0.30 m`。
- 平地足底净空目标：**`0.075 m`**。
- 楼梯足底净空目标：

\[
h_{\text{target}}=\max(0.17,\ h_{\text{step}}+0.02)
\]

示例：

| 台阶高度 | 足底净空目标 |
|---:|---:|
| `0.02 m` | `0.17 m` |
| `0.15 m` | `0.17 m` |
| `0.18 m` | `0.20 m` |
| `0.20 m` | `0.22 m` |

这是对参考项目 `0.17 m` 的 A2 高台阶扩展，尚未经过训练验证。

净空使用**脚当前位置下面的地表高度**，不是起跳支撑面。跨过台阶边缘时地表基准可以跳变，执行者不得自行改成平滑插值或提前看下一阶。

### 3.4 接触与腾空时间

每足独立维护：

```text
previous_raw_contact
has_contacted_once
air_time
```

规则：

1. `raw_contact = contacts.found AND foot_force_world_z > 5 N`。
2. `filtered_contact = raw_contact OR previous_raw_contact`。
3. 从未接触过地面的脚，第一次触地只初始化状态，不支付腾空奖励。
4. 脚已经接触过地面，随后无接触时，每步累计 `air_time += dt`。
5. 再次触地时，先用此前累计值计算 `(air_time-0.10)`，随后清零。
6. 持续站立不重复产生首次接触事件。
7. 部分环境 reset 时，只重置对应环境的状态。

数值验收：

- 腾空 `0.30 s` 后落地：单足奖励为 `+0.0008`。
- 腾空 `0.05 s` 后落地：单足奖励为 `−0.0002`。
- 连续站立：腾空奖励为 `0`。

### 3.5 前进奖励

采用：

```text
forward_progress_rate = (local_x_t - previous_local_x) / dt
reward_forward_progress = 0.4 * forward_progress_rate * dt
```

即每前进 `1 m` 得 `+0.4`，后退 `1 m` 得 `−0.4`。

- 不取绝对值。
- 不用历史最大位置代替实际位移。
- 不重复乘 `dt`。
- reset 后以新出生位置初始化 `previous_local_x`，禁止把传送算成进展。

### 3.6 相对 R2 删除或关闭的付款

以下旧奖励不再进入总奖励，系数统一为 `0`：

- 相对指令速度的分段 speed 公式。
- 单独超速、倒退、侧速分段罚项。
- 正向速度奖励的“双足安全支撑”门控。
- `footfall +0.5`。
- `swing +0.15`。
- `stall −0.25/s`。
- 原来的正向 pose 奖励。
- `angular_momentum`、`edge`、`soft_landing`、独立 `nonfoot` 奖励。
- 未在本 SDS 表中列出的其他奖励。

**接触、安全终止、首阶识别、停滞诊断仍保留。** 关闭付款不等于删除监控状态。

### 3.7 奖励设计的实际边界

新奖励允许爬楼俯仰，允许小幅竖直运动，并用连续净空误差代替大额中途事件。

但它仍有正的静止跟踪收益。例如 `c=0.3 m/s` 时，仅线速和偏航跟踪项，静止约为 `1.8465/s`。不能宣称这个公式已经排除停滞。

`γ=0.99` 下远处的登顶奖励折扣很强，`+10` 主要影响临近终点的行为。长楼梯学习依靠局部连续奖励和渐进课程，最终效果必须由真实训练验收。

---

## 4. PPO 配置

以第二个项目的楼梯 PPO 配置为主，保留本机可承受的批量规模。

| 参数 | R2 | v0.6 |
|---|---:|---:|
| 训练环境 | 512 | **512** |
| 评估环境 | 8 | **8** |
| rollout | 48 | **48** |
| epochs | 5 | **5** |
| minibatches | 4 | **4** |
| 初始学习率 | `5e−5` | **`3e−4`** |
| 自适应学习率范围 | `[1e−5,1e−4]` | **`[1e−5,3e−4]`** |
| desired KL | `0.01` | **`0.01`** |
| gamma | `0.995` | **`0.99`** |
| GAE lambda | `0.95` | **`0.95`** |
| PPO clip | `0.2` | **`0.2`** |
| entropy coefficient | `0.005` | **`0.003`** |
| value coefficient | `1.0` | **`1.0`** |
| clipped value loss | 开启 | **开启** |
| 全模型梯度裁剪 | `1.0` | **`1.0`** |
| 初始动作 std | `0.15` | **`0.30`** |
| std 范围 | `[0.10,0.30]` | **`[0.05,1.00]`** |

明确规则：

- 一个 Adam，所有可训练参数恰好出现一次。
- Adam：`betas=(0.9,0.999)`，`eps=1e−8`，`weight_decay=0`。
- minibatch KL `>0.02`：学习率除以 `1.5`。
- `0<KL<0.005`：学习率乘以 `1.5`。
- 之后限制在规定范围。
- 每次 Adam 更新后投影 std 到 `[0.05,1.00]`。
- 不增加双优化器、额外 Actor 冻结、KL 提前退出或自动回滚。
- 分别记录 Actor、Critic、融合层和 std 的裁剪前梯度范数；这些统计不改变梯度。
- 保留当前 advantage 标准化和超时 bootstrap 接线。

学习率、gamma、entropy 采用参考配置；std 与 rollout 是本机 ETH 迁移、资源条件下的适配值，不宣称已验证最优。

---

## 5. 课程表与训练预算

### 5.1 总流程

```text
ETH 行走初始化
  → F0 平地适应
  → H 六级楼梯逐步增高
  → D 逐步缩短踏面
  → L 增加楼梯长度
  → V 加入低速
  → R 扩大出生扰动
  → 冻结最终模型并验收
```

这是一条连续训练。阶段之间保留模型、优化器和训练账本，不重新初始化。

| 阶段 | 内容 | 最大 PPO 更新数 |
|---|---|---:|
| F0 | 平地适应 | 1,000 |
| H | 6 级楼梯，级高逐步从 2 cm 到 20 cm | 12,000 |
| D | 6 级、20 cm 高，踏深 25 cm → 20 cm | 3,000 |
| L | 20 cm 高、20 cm 深，12 → 20 → 40 级 | 6,000 |
| V | 40 级楼梯，加入 0.2 m/s 低速 | 4,000 |
| R | 40 级楼梯，扩大初始位置与航向扰动 | 4,000 |
| **总计** |  | **30,000** |

预算是上限，不要求用满。阶段提前通过即可前进，未用预算不转给其他阶段。

### 5.2 真实地形表

建立统一课程地形序列 `T`，索引从 0 开始：

| T 索引 | 名称 | 级数 N | 级高 h | 踏深 d |
|---|---|---:|---:|---:|
| `0…12` | `H00…H12` | 6 | `0.02 + 0.13*j/12`，`j=0…12` | `0.30` |
| 13 | H13 | 6 | `0.16` | `0.30` |
| 14 | H14 | 6 | `0.18` | `0.30` |
| 15 | H15 | 6 | `0.20` | `0.30` |
| 16 | D25 | 6 | `0.20` | `0.25` |
| 17 | D20 | 6 | `0.20` | `0.20` |
| 18 | L12 | 12 | `0.20` | `0.20` |
| 19 | L20 | 20 | `0.20` | `0.20` |
| 20 | L40 | 40 | `0.20` | `0.20` |

- `tile_id=0` 为平地。
- `T[k]` 对应 `tile_id=k+1`。
- 全部楼梯宽 `0.80 m`，踏板厚 `0.01 m`，顶部平台长 `2.0 m`。
- 仍为开放楼梯，不增加立板或护栏。
- 使用精确盒体几何，不按参考项目的高度场分辨率量化。
- 几何、地图、接触索引、奖励高度查询、成功判据和超时公式必须读取同一个 `TerrainSpec`。

### 5.3 环境混合

F0：512 个环境全部为平地。

其余阶段：

| 角色 | 环境数 |
|---|---:|
| 平地 | 102 |
| 当前最难地形 frontier | 164 |
| 临近难度 near | 123 |
| 简单地形 easy | 123 |

设当前 frontier 为 `T[k]`：

- frontier：全部使用 `T[k]`。
- `k≥2` 时，near 在 `T[k−2]、T[k−1]` 中等概率选择。
- `k<2` 时，near 使用 `T[k]`，与参考代码一致。
- easy 在 `T[0]…T[max(k−3,0)]` 中等概率选择。
- V、R 固定 `k=20`，因此 near 为 L12/L20，easy 为 `T[0]…T[17]`。
- 抽样仅发生在 episode reset。
- 角色分配和 terrain 抽样使用独立、可恢复的训练 RNG。
- 退级后按新 `k` 截断可采样集合，不能仍采历史最高难度。
- 只有 frontier 角色的 episode 能进入课程晋退统计，即使 near 恰好使用相同地形也不能混入。

其他环境数量只允许用于工程测试，采用最大余数法分配，平手顺序固定为 flat、frontier、near、easy。

### 5.4 指令与出生状态

用户已选择先上楼、后低速：

- F0/H/D/L：每回合 `c∼U[0.3,0.8] m/s`。
- `vy=0`，`wz=0`。
- 每回合内指令固定，不采用参考项目每 5 秒重采样。
- 不增加零指令驻站环境。

V 阶段前 1,000 次更新：

```text
p = min(stage_iteration / 1000, 1)
vx_min = 0.3 - 0.1*p
vx_max = 0.8 - 0.2*p
```

之后及整个 R 阶段：`c∼U[0.2,0.6]`。

F0/H/D/L/V 的出生范围：

```text
local_x ∈ [-1.10, -0.90] m
local_y ∈ [-0.05, +0.05] m
yaw     ∈ [-5°, +5°]
```

R 阶段前 400 次更新线性扩大到：

```text
local_x ∈ [-1.20, -0.80] m
local_y ∈ [-0.10, +0.10] m
yaw     ∈ [-10°, +10°]
```

R 的平地保留环境仍用原来的较小出生范围。

### 5.5 H 阶段的缓升快退

课程状态用整数 `level_tick∈[0,100]`，避免浮点边界歧义：

```text
level = level_tick / 100
frontier_row = floor(15 * level_tick / 100)
```

初值 `level_tick=0`。不复制参考项目的 `0.65` 初值。

H00 必须先连续两次通过完整门禁，才允许自适应升级；不要求额外固定训练 500 次。

每 **64 个当前 generation 的 frontier 完成 episode** 汇总一次：

- `S`：真实安全登顶比例。
- `F`：任务失败比例，包括侧向出界、bounds、机身碰撞、非足持续接触、姿态翻倒和下坠。
- `T`：超时未成功比例。
- `S+F+T=1`。
- `Q`：逐 episode 的线速原始指数得分均值，再对 64 个 episode 求均值；不包括偏航奖励、不乘权重。

数值异常直接停止，不当作正常训练样本继续统计。

EMA：

```text
第一批：EMA = batch_metric
后续：EMA = 0.97 * EMA + 0.03 * batch_metric
```

升级：

```text
S_EMA >= 0.65
Q_EMA >= 0.45
F_EMA <= 0.35
连续满足 5 次检查
且 cooldown == 0
→ level_tick += 1
```

退级：

```text
F_EMA >= 0.40 或 S_EMA <= 0.20
连续满足 2 次检查
→ level_tick -= 3
```

边界限制为 `[0,100]`。退级优先于升级，退级不受 cooldown 阻止。每次实际 level 改变后设置 `cooldown=5`，每次课程检查减一，最低为零。

实施细则：

- episode 按结束控制步、env_id 排序进入 FIFO。
- 每次 PPO 更新边界最多处理一个 64 例窗口。
- 未处理的完整或不足窗口保留，不丢弃。
- frontier row 改变时建立新 generation，清空 EMA、FIFO、ready/hard streak，并设置 cooldown 为 5。
- 旧 generation 未结束 episode 的结果保留在日志，但不进入新 generation 指标。
- 实际地形变化只在完整更新边界提交，并 reset 环境、接触、地图历史和奖励差分状态。
- `level_tick=100` 且 H15 连续两次完整门禁通过，才进入 D。

64 例窗口按本机 512 环境缩小；不照搬参考项目的 4096 例窗口。

### 5.6 D、L、V、R 晋退

D 子级预算：

```text
D25 = 1500
D20 = 1500
```

L 子级预算：

```text
L12 = 1500
L20 = 1500
L40 = 3000
```

D/L：

- 当前子级连续两次完整门禁通过才前进。
- 使用同样的 64 例窗口与 hard 条件；连续两次 hard 检查则退一个子级。
- 不跨宏阶段退级；最低子级保持当前级。
- 退级后建立新 generation，清 EMA、FIFO、门禁 streak 等状态。
- 历史通过记录不能直接用于再次升级，必须重新取得两次完整门禁。
- 每个子级预算累计使用，退级不归零。
- 返回已耗尽预算的子级，立即停止并报告预算失败。
- 不借用其他子级预算。

V：

- 至少完成 1,200 次更新。
- 门禁从第一天起就使用最终 `0.2/0.4/0.6 m/s`。
- 只有 V 更新数达到 1,000 后产生的门禁才累计晋级 streak。
- 连续两次通过后进入 R。

R：

- 至少完成 600 次更新。
- 所有门禁从第一天起使用最终出生扰动范围。
- 只有 R 更新数达到 400 后产生的门禁才累计晋级 streak。
- 连续两次通过后冻结最终模型。

预算最后一次更新必须先完成应执行的门禁，再判定结果。没有合法晋级条件则停止，不能以“还差一点”为由追加预算。

---

## 6. 门禁、退化监控与最终验收

### 6.1 完整门禁

| 阶段 | 评估速度 | 每速度例数 | 通过要求 |
|---|---|---:|---|
| F0/H/D/L | `0.3、0.5、0.8 m/s` | 20 | 每档至少 18 |
| V/R | `0.2、0.4、0.6 m/s` | 20 | 每档至少 18 |
| 最终冻结测试 | `0.2、0.4、0.6 m/s` | 100 | 每档至少 95 |

- 每 100 次更新执行完整门禁。
- H00/H15、D/L 的晋级门禁必须来自同一 generation，且该 generation 至少完成 100 次更新。
- 地形切换、退级后，旧门禁不用于累计新 streak。
- 评估使用确定性动作均值。
- 模型、归一化统计和优化器均不得在评估中更新。

楼梯门禁以**真实安全登顶**为准。精确速度误差继续报告，但本版不因超速单独否决登顶结果。

F0 每例观察 500 步，要求：

```text
无任务失败
max |local_y| <= 0.20 m
max |yaw_world| <= 10°
总前进距离 >= 0.5 * 指令速度 * 10 s
```

### 6.2 成功与安全终止

保留 R2 的真实平台接触成功定义：

- 四足在最近 50 控制步内均有顶部平台真实安全支撑。
- 单足支撑需连续 2 控制帧。
- 当前至少 2 足支撑。
- 机身进入顶部平台起点后至少 `0.8 m`。
- `|local_y|≤0.20 m`。
- 四足位置在平台安全内缩 `0.042 m` 内。
- 连续满足 5 控制帧。
- 没有任何 failure。

所有涉及 `N*0.20` 的计算改为 `N*当前tile踏深`。

保留机身接触、非足持续接触 `0.1 s`、姿态 `70°`、侧界、下坠、bounds、接触溢出和非数值保护。报告中将“侧向出界”与“物理翻倒”分开。

楼梯超时：

\[
n_{\max}=\left\lceil
\frac{2(2.2+N\sqrt{h^2+d^2})/c+10}{0.02}
\right\rceil
\]

平地训练上限保持 1,500 步。成功/失败不 bootstrap；超时使用 reset 前终端观测 bootstrap，GAE 不跨 episode 传播。

### 6.3 退化与停滞

每 50 更新执行 15 例诊断。

保留旧停滞检测，但只作为诊断与停机依据，不支付旧 stall 奖励。移除 swing/footfall 付款后，首阶识别必须继续有效。

新增保持集：

- F0 通过后，每 200 更新检查 15 例平地路线保持。
- H00 通过后，每 200 更新检查 15 例 H00 登顶保持。
- 每集三档初期速度各 5 例，固定 case。
- 同一保持集连续两次出现至少 `12/15` 失败，停止训练，状态 `behavior_regression`。
- 不因低速精确跟踪误差单独触发该停机。

当前宏阶段已训练至少 200 更新，且处于最低可退子级时，连续四个课程窗口 `F_EMA≥0.80`，停止。单独记录该计数，不能因一次无效退级尝试而清零。

### 6.4 最终测试

R 通过后：

1. 冻结当前 checkpoint 和 SHA。
2. 对最终 40 级目标楼梯执行三速各 100 例。
3. 每档至少 95 例安全登顶。
4. 任何一档不达标，结果记为未通过；不自动继续训练、不换 seed、不换历史最佳模型。
5. 保存三档视频、完整逐例结果和失败案例列表。
6. 单独报告速度跟踪结果，不能把“登顶通过”写成“精确速度跟踪通过”。

---

## 7. 文件修改与模块接口

### 7.1 新增独立生产线

新增目录：

```text
src/tasks/velocity/v06/
```

从 R2 复用结构，新增或调整以下模块：

| 文件组 | 必须实现的内容 |
|---|---|
| `spec.py` | 唯一数值源、奖励表、PPO、课程地形表、预算、身份与协议 |
| `terrain.py`、`env_cfg.py`、`map.py` | v06 真实地形、地表查询、碰撞索引和高程图接线 |
| `contacts.py`、`events.py` | 与新 tile 接线的接触/登顶识别；旧事件仅作诊断 |
| `rewards.py`、`env.py` | 新奖励、腾空状态、真实执行目标、reset 前逐环境快照 |
| `curriculum.py` | 课程状态、FIFO、EMA、晋退、generation、子级累计预算 |
| `model.py`、`ppo.py` | ETH 迁移、新 std/PPO 配置、梯度分组统计 |
| `runner.py`、`recovery.py` | 单条训练、schema8、完整更新边界恢复、资格与最终验收流程 |
| `evaluation.py`、`speed.py` | 新速度组、固定 case、保留集、最终门禁 |
| `orchestration.py` | 准入、进程串行、暂停和停止状态 |
| `telemetry.py`、`connection.py` | 新指标、W&B 接线、沟通与资源状态 |

**重要：** 不得保留隐式绑定旧 `v02.spec.STAGES` 的地形或地图路径。仅改 v06 的表，不会自动改变旧环境生成的几何。

允许复用与课程无关的底层数学、网络组件；禁止修改旧版本共享行为来实现 v06。

### 7.2 必须固定的接口

```text
TerrainSpec:
  tile_id, lesson_id, N, h, step_depth,
  width, tread_thickness, platform_length,
  frame_origin_world, valid_bounds_local

RewardFrame:
  executed_command, pre_reset_body_state,
  q, qdot, q_target_executed, action,
  foot_position, foot_velocity, foot_force,
  tile_metadata, success, failure

RewardResult:
  raw_terms[name]       float[B]
  weighted_terms[name]  float[B]
  total                float[B]

CurriculumState:
  phase, substage, level_tick, generation,
  generation_updates, spent_by_phase, spent_by_substage,
  episode_fifo, ema, ready_streak, hard_streak,
  cooldown, gate_streak, retention_streaks,
  minimum_level_failure_streak, sampler_rng_state
```

这些状态必须能够序列化。训练恢复不得丢掉 EMA、预算或抽样 RNG。

### 7.3 新入口

新增对应 v06 版本：

```text
scripts/a2_v06_bootstrap.py
scripts/train_a2_ideal_stairs_v06.py
scripts/run_a2_ideal_stairs_v06.py
scripts/evaluate_a2_ideal_stairs_v06.py
scripts/play_a2_ideal_stairs_v06.py
scripts/smoke_a2_ideal_stairs_v06.py
scripts/test_a2_ideal_stairs_v06.py
scripts/check_a2_ideal_stairs_v06_migration.py
scripts/diagnostics/analyze_a2_v06_rewards.py
tests/test_v06_*.py
```

所有入口先执行 v06 bootstrap，再加载任务，避免旧任务扫描产生隐式配置。

---

## 8. 证据、恢复与工程纪律

### 8.1 训练账本

每个正常 PPO 更新：

```text
控制样本 = 512 × 48 = 24,576
物理子步 = 98,304
Adam 更新 = 5 × 4 = 20
```

30,000 更新上限对应：

```text
737,280,000 控制样本
2,949,120,000 物理子步
600,000 Adam 更新
```

评估、视频、smoke、probe 单列成本，不混入训练账本。

### 8.2 checkpoint

schema8 至少保存：

- 模型、单 Adam 状态、当前学习率和归一化状态。
- 全部学习计数与课程状态。
- 当前/待提交阶段转换事务。
- 评估 case 表及已完成门禁身份。
- Python、NumPy、Torch、CUDA、环境和地形抽样 RNG。
- SDS、配置、源码闭包和初始化源 SHA。
- 明确 `resume_mode=new_episode`。

恢复后开始新 episode，不声称物理轨迹无缝或逐位复现。已计费学习更新不得重跑。

拒绝用 R1/R2、其他配方、smoke/probe checkpoint 作为正常 v06 训练恢复来源。

### 8.3 评估隔离与 D1 防回归

- 仅在 case 构造层添加一次 seed offset。
- validation：`seed+200300`。
- diagnostic：`seed+500300`。
- perturbed：`seed+300300`。
- final test：固定 `400342`。
- runner 传原始训练 seed，不预先加 offset。
- 主路径、恢复重放、保持集都必须测试。

评估前后比较可序列化训练状态：模型、归一化器、优化器、RNG、环境缓存及计数。不能再把 `statistics_unchanged` 写死为 true，也不能据此宣称引擎逐位可复现。

### 8.4 逐帧证据

记录每个评估环境 reset 前的：

- 机身位置、朝向、机体/世界系速度。
- 执行指令与实际关节目标。
- 四足位置、速度、接触力、支撑面 ID。
- 逐环境奖励分项与总和。
- 成功、失败、超时和所有原因 mask。
- 首阶、停滞、课程 tile 和 generation。

禁止把全批环境平均奖励当成某个案例的奖励。

### 8.5 保存与资源

- 每 100 更新保存 checkpoint；门禁、阶段转换、异常停止也保存。
- 训练诊断每 50 更新。
- 一次只允许一个 GPU worker。
- 保持整卡 `≤7168 MiB`、训练进程 `≤5120 MiB`、主机可用内存 `≥4096 MiB`、磁盘空闲 `≥10 GiB`。
- 资源不足时停止并上报，不自动减环境数或 rollout。
- 既有 W&B 监控超时策略保持，不在本版暗改。
- 用户停止标记为 `user_requested_stop`，不能再归类为监控故障或门禁失败。

---

## 9. 验证清单

### 9.1 CPU 与纯函数

必须覆盖：

1. 奖励分项之和等于总奖励；误差 `≤1e−5`。
2. 连续项 `dt` 只乘一次；终端项不乘 `dt`。
3. 前进 `0.01 m` 得 `0.004`，不受重复 dt 缩放影响。
4. 腾空 `0.30/0.05 s` 的两个精确数值用例。
5. 首次出生接触、持续站立、单帧失联、部分 reset 不重复付款。
6. `v_z=0.15` 时竖直项为零；`v_z=0.25` 时速率为 `−0.01`。
7. 纯俯仰不会产生 roll 惩罚；横滚产生预期惩罚。
8. 脚跨台阶边缘前后，净空高度使用当前位置地表；锁定跃迁真值。
9. `previous_x/action/qdot` 不跨 episode 差分。
10. 实际执行的关节目标用于 joint tracking。
11. 22 个 tile 的真实几何、metadata、地图查询、平台与边界一致。
12. 不同踏深不再被旧 `.20` 常量覆盖。
13. H tick 边界、64 例 FIFO、EMA 首批初始化、cooldown、升降优先级。
14. 旧 generation 结果不会驱动新课程。
15. 退级不重置子级预算，不复用旧通过。
16. V/R 门禁与训练 ramp 解耦。
17. 成功与 failure 同帧时 failure 优先。
18. seed offset 只添加一次。
19. 评估修改训练状态的反例必须被检测。
20. schema8 拒绝旧 checkpoint 和工程测试容器。
21. 用户停止状态与训练失败、通信失败分离。
22. R2 历史文件 SHA 不因 v06 实施变化。

反例分析必须包括：静止、正常前进、后退、横向漂移、超速、抬腿不落地、重复踏步、正常上阶、摔倒、临近终点拖延。分别报告未折扣和与 `γ=.99` 折扣和；缺少真实状态的项明确标为未计入。

### 9.2 ETH 迁移

- 校验源 SHA。
- Actor 所有源张量按清单迁移。
- 合成输入上的动作均值与 ETH 误差 `≤1e−6`。
- 融合旁路初始为零。
- 新 std 恰为 `0.30`。
- Critic 和 Adam 为新状态。
- 不能用随机采样动作一致作为均值迁移验收。

### 9.3 GPU 工程检查

先执行 smoke：

```text
seed42、8环境、H00、2更新
rollout48、epochs5、minibatches4
768 控制样本、3072 物理子步、40 Adam
```

再执行一次资源与恢复 probe：

```text
512训练环境 + 8评估环境
5次预热 + 50次测量 + 1次恢复更新 = 56次学习更新
1,376,256 控制样本
```

probe 使用最终地形验证资源和恢复；评估、无学习测试成本另记。smoke/probe 都标记为不可用于正式训练恢复。

任何失败先定位修复。不得把工程检查通过写成“奖励有效”或“已学会上楼”。

---

## 10. DSH 工作流程与验收分工

### W0：冻结执行合同

- [ ] 保存本 SDS、参考审计和执行提示词。
- [ ] 计算 SDS 与参考源 SHA，生成 manifest。
- [ ] 核实 R2 seed43 仍停止，旧证据保留。
- [ ] 建立 `logs/a2_ideal_stairs_v06/implementation/coordination/`。
- [ ] 追加 `logs/codex_changes.md`。
- [ ] 冻结模块接口和文件 owner，随后才并行修改。

### 执行子 agent

| 子 agent | 工作 |
|---|---|
| A | `spec/rewards`、奖励状态、离线反例、奖励测试 |
| B | 地形、地图、接触、env、新课程和采样 |
| C | 模型迁移、PPO、runner、恢复、评估、入口与监控 |

DSH 主对话负责跨模块接线、冻结 SHA、核对预算、汇总问题和运行次序。不得由多个 owner 同时修改同一文件。

### 独立验收子 agent

不验收自己实现的模块，必须：

- [ ] 按 SDS 逐项复算参数和公式。
- [ ] 检查真实地形，而非只检查配置文本。
- [ ] 检查 terminal/reset 顺序、seed、评估隔离和恢复。
- [ ] 独立复核 CPU、迁移、smoke、probe 证据。
- [ ] 对同一冻结源码 SHA 出具 `V06_IMPLEMENTATION_ACCEPTANCE.md`。
- [ ] 明确 blocking 数量；blocking 必须为 0 才允许开始训练。

### 自动训练顺序

```text
实施
→ CPU/迁移验收
→ GPU smoke
→ 资源与恢复 probe
→ 独立验收通过
→ 新建 v06 seed42 run
→ F0/H/D/L/V/R 连续训练
→ 最终冻结测试
→ 视频与工作汇报
```

不再执行“两个 pilot 通过后重新初始化 formal”。

发生训练失败、预算耗尽、退化停机或未覆盖问题时：

1. 在完整更新边界保存证据并停止。
2. 写 `Qxxx.md`，包含问题、可复现证据、候选处置和影响。
3. 通知本 Codex 裁决。
4. 不换 seed 抽签，不加预算，不改门禁，不自动选旧 checkpoint 继续。
5. 不阻塞的独立 CPU 工作可以继续，依赖裁决的工作等待。

---

## 11. 执行入口合同与交付物

### 11.1 CPU 测试命令

实施后提供：

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2

CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/xxs/anaconda3/envs/a2/bin/python \
  scripts/test_a2_ideal_stairs_v06.py \
  --pattern 'test_v06_*.py' \
  --maxfail 0
```

### 11.2 训练命令接口

以下参数必须作为配方一致性断言实现，不允许用它们绕过 SDS：

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2

MUJOCO_GL=egl \
/home/xxs/anaconda3/envs/a2/bin/python \
  scripts/run_a2_ideal_stairs_v06.py \
  --mode train \
  --execute \
  --recipe 0.6-go2refs-r1 \
  --seed 42 \
  --device cuda:0 \
  --num-envs 512 \
  --eval-num-envs 8 \
  --rollout 48 \
  --epochs 5 \
  --minibatches 4 \
  --learning-rate 0.0003 \
  --gamma 0.99 \
  --entropy-coef 0.003 \
  --initial-std 0.30 \
  --std-min 0.05 \
  --std-max 1.00 \
  --initial-vx-min 0.30 \
  --initial-vx-max 0.80 \
  --final-vx-min 0.20 \
  --final-vx-max 0.60 \
  --vy 0.0 \
  --wz 0.0 \
  --max-updates 30000 \
  --phase-budgets F0=1000,H=12000,D=3000,L=6000,V=4000,R=4000 \
  --save-interval 100 \
  --diagnostic-interval 50 \
  --gate-interval 100 \
  --init-checkpoint /home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt \
  --init-sha256 8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722 \
  --acceptance-file logs/a2_ideal_stairs_v06/implementation/V06_IMPLEMENTATION_ACCEPTANCE.md \
  --output-dir logs/a2_ideal_stairs_v06/train_seed42_trial01
```

- 无 `--execute` 时只预检，不启动 GPU 或创建目标 run。
- 输出目录已存在则拒绝，禁止覆盖。
- 验收报告必须绑定当前源码、SDS 和配置 SHA。
- 工程验收通过后，DSH 按用户既有授权自动启动上述新训练，不再额外询问是否训练。
- 恢复必须使用明确的 `--resume-run` 接口，并校验原合同；不能重新走 ETH 初始化。

### 11.3 最终交付

- 完整 SDS、参考审计、独立验收报告。
- 全部 v06 代码与测试。
- 奖励反例表、迁移报告、资源报告。
- 单条训练各阶段消耗、晋退记录、checkpoint SHA。
- 三档最终测试逐例结果与视频。
- 失败时的具体停机阶段、原因和证据。
- `logs/codex_changes.md` 的追加记录。

工作汇报必须说明实际修改、关键参数、训练流程、已通过检查和未验证事项。只有最终训练数据支持时，才能称为效果改善。

---

## 12. 给 DSH 的执行提示词正文

> 执行项目 `/home/xxs/dog/unitree_rl_mjlab_a2` 的  
> `doc/sds/SDS_A2_IdealHeightMap_Stairs40_v0.6.md`。
>
> 这是替代此前未实施 0.6 草案的新合同，奖励和课程以两个固定版本 Go2 项目的源码为主要依据。所有公式、数值、预算、门禁和适配差异按 SDS 实施。
>
> 你作为主对话负责协调，启动 A/B/C 执行子 agent，并另设独立验收子 agent。先冻结接口和文件 owner，再并行工作；验收 agent 不审核自己写的模块。
>
> 新增隔离的 v06 产品线，不修改 R1/R2 的配方、运行目录、停止文件和历史证据。R2 seed43 保持停止。
>
> 严格按 W0、CPU/迁移、GPU smoke、probe、独立验收、单条 seed42 训练、最终验收的顺序执行。独立工程验收 blocking=0 且 SHA 校验通过后，自动启动 v06 seed42；不再跑两个 pilot 后重新开始 formal。
>
> 遇到 SDS 未覆盖、互相冲突、无法实现或测试暴露的设计问题，写入 coordination/Qxxx.md 并向本 Codex 提问。你负责执行和提供证据，配方与协议裁决由本 Codex 负责。不得自行改权重、速度、课程、PPO、预算、门禁或换 seed。
>
> 训练失败或退化时保存完整证据并停止，不自动重试、不挑选历史模型替代终点。
>
> 每次项目修改追加 `logs/codex_changes.md`。完成后提供实现说明、关键参数、验证结果、训练进度和对应文件链接；不得把工程通过表述为训练效果通过。
