# A2 理想高程图镂空楼梯 0.5-r2 实施与运行文档

- 代码版本：`0.5.1`；奖励配方：`0.5-r2`；checkpoint schema：**`7`**；architecture：**`v05-route-state-v1`**。
- 正式规范：`doc/sds/SDS_A2_IdealHeightMap_Stairs40_v0.5_r2.md`（修订 1.0，1095 行），SHA256 `9ff356b022e6205f0a181ff4048ef0f6e773742250d3174bf9b087986c8a4abc`。
- 实施裁决：`logs/a2_ideal_stairs_v05/implementation/coordination/Q003_r2_implementation_decision.md`，SHA256 `0cd996e72c494a6cd981db067230710dbc8a5c0930d97bed1e86069a86895bf1`。
- 设计裁决：`logs/a2_ideal_stairs_v05/implementation/coordination/Q003_r2_design_decision.md`，SHA256 `78515bb951a96a8c69b9f10b90043a6312ed39124bc7898bd08b89e046d3a0fc`。
- 任务身份：`Unitree-A2-IdealHeightMap-Stairs-V05R2`；楼梯协议 `v05-r2-stair-climb-first-v1`；F0 协议 `v05-r2-flat-world-route-v1`；日志根 `logs/a2_ideal_stairs_v05_r2/`。
- 本文是运行文档，**不是训练报告**。它只记录真实存在的接口、冻结数值、真实命令模板和已执行/未执行项。**不得**用本文声称任何爬楼成功率、停车改善或部署就绪。

本文覆盖 0.5-r2 的入口脚本、W2 反例复验、分析测试与本运行文档：

| 文件 | 归属 |
|---|---|
| `scripts/{train,run,evaluate,play,smoke,test}_a2_ideal_stairs_v05_r2.py` | 本轮 |
| `scripts/diagnostics/analyze_a2_v05_r2_rewards.py` | 本轮（W2） |
| `tests/test_v05_r2_reward_analysis.py` | 本轮 |
| `docs/ideal_heightmap_stairs_v05_r2.md` | 本轮（本文件） |

`src/tasks/velocity/v05_r2/*`、`scripts/a2_v05_r2_bootstrap.py`、`scripts/check_a2_ideal_stairs_v05_r2_migration.py` 由其他工作包拥有；本文只记录其真实接口。

## 0. 已执行 / 未执行（一次说清）

| 项目 | 状态 | 证据 |
|---|---|---|
| 六个入口脚本 + 分析脚本的真实 `--help` | **已执行**（`CUDA_VISIBLE_DEVICES=`） | §8.1；全部 exit 0 |
| W5 预检 pilot42 / pilot43（无 `--execute`） | **已执行**：两次 exit 0、`status=preflight_validated`、`total_budget=update_cap=2500`、`binding_sha256=0fef47ee…`、目标 run 目录**未创建**、`training_started=gpu_started=network_started=false` | `implementation/preflight/preflight_pilot_seed4{2,3}.json` |
| W7 形态预检 probe（无 `--execute`） | **已执行**：exit 0、`total_budget=update_cap=56`、`probe_learning_updates=56`、目标目录**未创建** | `implementation/preflight/preflight_probe_seed42.json` |
| `--resume-run` 对 probe 的拒绝 | **已执行**：exit 2，`W7 resource probe is a one-shot resource run; it is not resumable` | §8.2 |
| formal 预检 | **未执行**：formal 需要两个真实通过的 pilot 报告；本轮用负向检查证明缺报告即拒绝（exit 2） | §8.2 |
| W2 反例 + 锚点本地复验（r2 纯函数/状态机） | **已执行**：全部必需检查通过 | `logs/a2_ideal_stairs_v05_r2/implementation/analysis/a2_v05_r2_reward_analysis.{json,md}`、`analysis_index.json`（§8.5） |
| 分析脚本不变量测试 `tests/test_v05_r2_reward_analysis.py` | **已执行**：11 passed | §8.4 |
| r2 CPU 测试全套 `--pattern 'test_v05_r2_*.py'` | **已执行**（实际计数见 §8.3） | §8.3 |
| worker CLI 接线（`Supervisor.worker_command()` → `train` 解析+校验） | **已执行**：4 条接受（pilot42/pilot43/formal42/formal42-resume）、6 条拒绝（均 exit 2） | `implementation/wiring_checks/worker_cli_validation.json`（§8.7） |
| evaluate / play 的 fail-closed 负向检查 | **已执行**（`--num-envs`、错误 SHA、`--max-steps` 过短、已存目录、initialization 容器拒绝） | §8.8 |
| ETH 迁移审计 seed42 / seed43（A2 包） | **已执行**（他人执行，本轮只读取）：两份 `status=passed`、`max_abs_error=0.0`、绑定 73 个源文件 | `implementation/migration_seed42/`、`migration_seed43/`（§8.6） |
| CPU 工程 smoke（`--device cpu`） | 见 §8.9 的实际结果（只能标 `passed_cpu_engineering_only`） | §8.9 |
| W6 GPU smoke（`--device cuda:0`） | **未执行**：唯一 GPU worker 属于 r1 seed43，本轮 CPU only | — |
| W7 资源 probe / W8 两个 pilot / W9 formal / W10 final test + 三速播放 | **未执行**（probe 只做了无 `--execute` 预检形态） | — |
| evaluate / play 的真实仿真 | **未执行**：需要真实 schema7 checkpoint；本轮**没有**伪造 checkpoint 充当证据（仅做 fail-closed 负向检查） | §8.8 |
| commit / push | **未执行**（按纪律不 commit） | — |

**源码闭包会随并行编辑失效**：schema7 的 `contract_sha256` 绑定全部 v0.5-r2 生产文件与入口脚本的静态闭包。任何一次后续编辑都会使此前生成的 `migration_zero.pt`、W2 证据与 checkpoint 在 `preload_checkpoint` 处被判为 contract/source closure 不一致。因此**冻结后必须重新生成受影响证据**；本文记录的 SHA 只是"生成当时"的事实。

## 1. 版本身份与冻结参数

| 项 | r2 固定值 |
|---|---|
| `CODE_VERSION` / `RECIPE_VERSION` / `SDS_REVISION` | `0.5.1` / `0.5-r2` / `1.0` |
| `SCHEMA_VERSION` | `7` |
| `ARCHITECTURE` | `v05-route-state-v1` |
| `TASK` | `Unitree-A2-IdealHeightMap-Stairs-V05R2` |
| `PROTOCOL_ID` / `FLAT_PROTOCOL_ID` | `v05-r2-stair-climb-first-v1` / `v05-r2-flat-world-route-v1` |
| `LOG_ROOT` | `logs/a2_ideal_stairs_v05_r2` |
| 唯一初始化权重 | `/home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt` |
| 初始化 SHA256 | `8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722` |
| 观测键（五键，顺序固定） | `proprio[45]`、`history[5,45]`、`heightmap[3,48,32]`、`privileged[35]`、`route_error[3]` |
| normalizer | Actor 用 FrozenETH 45 维（冻结）；Critic 80 维（`proprio+privileged`）；`route_error` **不进任何 normalizer** |
| 控制 | `physics dt=0.005 s`、`decimation=4`、`control dt=0.02 s`、地图 10 Hz |
| 环境数 | 训练 512、评估 8、播放 1；`device=cuda:0` |
| smoke 容器 | `v05-r2-smoke-state-v1`，`formal_resumable=false` |

## 2. `route_error` 契约与 `ideal_route_state` 部署依赖

### 2.1 精确语义（SDS §4.2）

对每个环境 `i`、每个控制步：

```text
tile       = tile_ids[i]                                  # 该环境槽位的实际 atlas tile
origin     = origins[tile]                                # 与奖励/门禁同源的 frame origin
local      = root_link_pos_w[i] - origin                  # 与 _task_state 的 local 同一表达式
local_y    = local[1]
yaw_world  = euler_heading(root_link_quat_w)[2]           # 弧度
yaw_axis   = 0.0                                          # 当前 atlas 各 tile 只平移
e_world    = wrap_to_pi(yaw_world - yaw_axis)             # atan2(sin,cos)，弧度
route_error = [clip(local_y / 0.20, -4.0, 4.0),           # 门禁横向阈值的 5 倍缩放
               sin(e_world),
               cos(e_world)]                              # float32[B,3]
```

- 第一维：`local_y=0 → 0`；`|local_y|=0.20 m → ±1.0`；`|local_y|≥0.80 m → 饱和 ±4.0`（SDS §4.2、§9.3-21；W2 证据见 `analysis` 的 `route` 块）。
- 第二/三维：`e_world` 的单位向量；对齐时 `(0,1)`；用弧度，不用度，不用 `yaw_world` 本身，不用出生线 `e_perp`。
- **per-tile、同帧**：每环境用**自己**的 tile 与**同一帧**位姿；不得按 supervisor 当前 stage 给整批取同一 tile。
- **F0（N=0）与 P/S 同一公式**：flat tile 0 也有自己的 origin，没有特殊分支、不补零。
- **不归一化、不加噪声、不 clamp 到 [-1,1]**（clip 只发生在第一维公式内部）；**不进 ETH/Critic normalizer**，无 running mean/var。
- 缺失/形状不符/dtype 非 float32/device 不符/batch 不一致/非 finite：**硬错误，禁止补零**（SDS §4.7）。
- `reset` 后由新位姿重算（post-reset）；`info['terminal_observation']['route_error']` 必须是 **reset 前**同帧值；同一步 reset 槽的返回 `obs` 是 reset 后新回合值，两者必须不同（SDS §4.5）。

### 2.2 消费方

| 消费方 | 必须使用 | 禁止 |
|---|---|---|
| `PPO.collect` | 环境返回的完整五键观测；`storage['obs']['route_error']` 为 `[24576,3]` | 删键、重建 obs、补零 |
| `model.forward` / `action_distribution` | `proprio/history/heightmap/route_error` | 缺 route 时默认零 |
| `model.evaluate` | 五键，Critic 末尾 3 维 = `route_error`（索引 208:211） | 只喂 208 维或截断 route |
| `terminal_observation` | 同一帧 pre-reset 五键 | 用 reset 后观测冒充终端 |
| evaluation | 由 r2 env 构造的五键；原始 NPZ 记录 `route_error[steps,3]`/`local_y`/`yaw` | 复用 r1 obs、手工造 route |
| play | 由 r2 env 构造的五键 | 从视频/日志/IMU 反推 route |
| runner / checkpoint | 只持久化模型/学习/课程 | 把 rollout 里的 route 当状态 |

### 2.3 `route/*` 只读诊断（SDS §4.8）

`route/abs_y_mean`、`route/abs_y_max`、`route/normalized_abs_y_max`、`route/saturation_rate`、`route/heading_deg_mean`、`route/heading_deg_max`、`route/nonfinite_count`（每更新聚合，`updates.jsonl`）；`route_final`、`route_max_abs_y`、`route_max_abs_heading_deg`、`route_initial`（每回合，`episodes.jsonl`，无命名空间）。

- `route/nonfinite_count` 必须为 `0`；非 0 是**硬错误**，不是指标。
- `route/*` 归入既有第 4 组面板 `运动/接触`；`panel_count=8` 不变，本版**不新增第 9 组**。
- `route/*` 不得出现在 `ppo/*`、`reward/*`、`eval/*` 的通过判据中，也不得回灌观测/normalizer/奖励/advantage。

### 2.4 部署依赖 `ideal_route_state`（本版新增，**无硬件验证**）

- Actor 与 Critic 都消费 `route_error`，因此真实部署必须提供**同语义**的世界/tile 系横向位置 `local_y` 与航向参考 `e_world`（定位/里程计/感知到的走廊或楼梯轴向），部署接口里标为 **`ideal_route_state`**。
- 它**不是**普通 IMU 量：不得用 IMU 积分自动替代横向位置（会漂移），不得用零值/上一帧缓存伪造，缺失必须拒绝运行策略。
- 当前**没有任何硬件部署验证**；仿真中的 tile 参考是已知量（atlas 只平移、`yaw_axis=0`）。坐标系标定、定位误差、失效降级均未验证。
- 因此**不能宣称部署依赖不变**；r1 的部署输入不含路线状态，r2 是新增依赖。真实部署方案属 SDS §13 未决问题。

## 3. 网络：Actor 旁路与 211 维 Critic

### 3.1 Actor（新增唯一旁路）

```text
features = concat(map_encoder(heightmap) -> 64, history_encoder(normalizer_eth(history)) -> 64)   # 128
h = ELU( actor.mlp[0](normalizer_eth(proprio))        # Linear(45,512)，ETH 迁移
       + actor.fusion(features)                       # Linear(128,512)，ETH 迁移时为零
       + actor.route_fusion(route_error) )            # 新增 Linear(3,512)，恒零初始化
mean = actor.mlp[2:](h)                               # 512→256→128→12
```

- 新增参数**只有** `actor.route_fusion.weight [512,3]` 与 `actor.route_fusion.bias [512]` 两个 tensor；先构造（消耗 RNG）再显式清零，不跳过构造。
- 初始化保持 ETH 动作均值：128 组合成观测上 `max_abs_error = 0.0`（迁移审计实测）；`route_error`、map、history 单独变化都不改变零融合输出。
- `std` 12 维全部写成 `0.15`（不继承 ETH std）；`[0.10,0.30]` 双处约束每次 Adam 后投影；均值无 tanh/clip。

### 3.2 Critic（208 → 211，从头初始化）

```text
raw80     = critic_normalizer(concat(proprio, privileged))            # [B,80]，mask[48:52] 仍 False
history_n = (history - critic_normalizer.mean[:45]) / var[:45].sqrt().clamp_min(1e-5) -> clamp(-5,5)
input211  = concat(raw80, critic_map_encoder(heightmap) -> 64,
                   critic_history_encoder(history_n) -> 64,
                   route_error -> 3)                                  # 末尾，208:211
value     = critic_mlp(211 -> 512 -> 256 -> 128 -> 1)
```

- `critic_normalizer` 仍 80 维，`route_error` 不参与统计；`update_critic_statistics` 仍只在每次 rollout 收集前调用一次、只喂 80 维。
- Critic 不迁移 ETH、不复制 r1 Critic；与同 seed、同构造顺序的 fresh 参考逐 tensor 相等。

### 3.3 ETH 迁移七步（`V05R2ActorCritic.from_eth`）

CPU 校验文件 SHA/ETH tensor 键形状 dtype finite/normalizer 合法 → 设定 run seed → 构造 r2 网络 → 完整复制 ETH Actor MLP 与冻结 normalizer → map/history encoder、`fusion`、`critic*`、`route_fusion` 按 seed 新建并把 4 个 fusion tensor 清零 → `std=0.15` → Critic 统计 fresh、Adam 空、计数 0 → 保存 `migration_zero.pt`（`scope=initialization`，不是 resume checkpoint）。

## 4. flat 初态：B1 修正与 B2 新增（两件事分别声明）

适用于**所有实际 `N=0` 的 flat 槽**：F0 的全部 512 槽，以及 P1～S5 混合槽中的 flat 槽（P1 的 102、P2～S5 各 51）。

```text
x    ~ U[-1.1, -0.9]   m      # 中心 x=-1（旧 metadata 的 spawn_x=0 只读使用，不改 v02/terrain.py）
y    ~ U[-0.05, 0.05]  m
yaw  ~ U[-5°, +5°]     rad    # U[-pi/36, +pi/36]
```

| 变更 | r1 实际 | r2 | 影响 |
|---|---|---|---|
| **B1 flat 出生 x 修正** | `x=0` | `x` 中心 **-1** | 平地与混合槽 flat 回合多约 1 m 接近段；F0 的 x 分布与评估表对齐 |
| **B2 flat 初态随机化（新增）** | `y=0`、`yaw=0` 确定性 | `y~U[-0.05,0.05]`、`yaw~U[-5°,5°]` | 训练初态首次覆盖评估 F0 的 y/yaw 范围；**r1 从未随机化 flat 初态** |

- 采样顺序固定 `x → y → yaw → command`；子集内按 `ids` 升序；S5 当前槽（r1 ramp）与 flat 槽并存时**先 flat 子集，再 N>0 子集**；只用该环境既有 `self.rng`。
- `initial_cases`（评估/播放）优先，**不消耗** RNG。
- `episodes.jsonl` 记录真实 `initial_pose`、`initial_sample_order='x,y,yaw,command'`、`rng_state_sha256_before_sample`、`initial_pose_source ∈ {r2_flat_uniform, r1_fixed, r1_s5_ramp, initial_cases}`。
- N>0 的 P1～S4 当前/previous 槽仍固定 `[-1,0,0]`；S5 当前槽仍按 400 更新 ramp 线性扩展到 `x±0.2`、`y±0.1`、`yaw±10°`。

## 5. 完整数值配置

### 5.1 阶段表（tile / N / h / 预算 / 完整门禁间隔）

| stage | tile_id | N | h(m) | 更新上限 | 门禁间隔 |
|---|---:|---:|---:|---:|---:|
| F0 | 0（flat） | 0 | 0.00 | 2000 | 100 |
| P1 | 1 | 1 | 0.02 | 1000 | 100 |
| P2 | 2 | 1 | 0.04 | 1000 | 100 |
| P3 | 3 | 1 | 0.06 | 1000 | 100 |
| P4 | 4 | 1 | 0.08 | 2000 | 100 |
| S1 | 5 | 4 | 0.08 | 3000 | 200 |
| S2 | 6 | 8 | 0.12 | 4000 | 200 |
| S3 | 7 | 12 | 0.16 | 4000 | 200 |
| S4 | 8 | 20 | 0.20 | 5000 | 200 |
| S5 | 9 | 40 | 0.20 | 7000 | 200 |

### 5.2 预算（单一来源 `spec.mode_budget`）

| mode | total / update_cap | 阶段表 | 说明 |
|---|---:|---|---|
| `pilot` | **2500** | `F0=2000, P1=500` | P1 固定跑满 500，不提前晋级 |
| `formal` | **30000** | 十阶段全表 | 737,280,000 控制样本 / 600,000 Adam |
| `probe`（W7） | **56** | 借用 formal 表（只为配置一致） | 5 warmup + 50 measured + 1 recovery；评估/S5 无学习探针成本**另列** |
| `smoke`（W6） | **2** | 不在 `MODE_BUDGETS` 中 | 40 Adam / 768 控制样本 |

- worker、阶段机、supervisor、resume 四处读取同一 `mode_budget(mode)`；`result.json` 的 `total_budget == update_cap == 有效上限`，`effective_cap_source='mode_budget'`。
- worker 的 `--max-updates` **必须等于**该 mode 的有效上限，传其他值直接拒绝（不再 clamp、不再出现 r1 的 `29500` 十阶段求和缺陷）。
- r1 `result.json` 的 `29500` 保留为历史事实，不追改；r2 报告必须写明"r1 记录是历史元数据缺陷，r2 已用单一来源修正"。

### 5.3 槽位（固定，不允许 allocator 追加）

| stage | 512 环境 flat/previous/current | 8 环境 flat/previous/current | 8 环境 tile |
|---|---|---|---|
| F0 | 512 / 0 / 0 | 8 / 0 / 0 | `[0,0,0,0,0,0,0,0]` |
| P1 | 102 / 0 / 410 | 2 / 0 / 6 | `[0,0,1,1,1,1,1,1]` |
| P2 | 51 / 102 / 359 | 1 / 1 / 6 | `[0,1,2,2,2,2,2,2]` |
| P3 | 51 / 102 / 359 | 1 / 1 / 6 | `[0,2,3,3,3,3,3,3]` |
| P4 | 51 / 102 / 359 | 1 / 1 / 6 | `[0,3,4,4,4,4,4,4]` |
| S1 | 51 / 102 / 359 | 1 / 1 / 6 | `[0,4,5,5,5,5,5,5]` |
| S2 | 51 / 102 / 359 | 1 / 1 / 6 | `[0,5,6,6,6,6,6,6]` |
| S3 | 51 / 102 / 359 | 1 / 1 / 6 | `[0,6,7,7,7,7,7,7]` |
| S4 | 51 / 102 / 359 | 1 / 1 / 6 | `[0,7,8,8,8,8,8,8]` |
| S5 | 51 / 102 / 359 | 1 / 1 / 6 | `[0,8,9,9,9,9,9,9]` |

- `previous` 槽使用**紧邻前一阶段自己的 N/h**（S1 的 previous = P4），不是 supervisor 当前阶段。
- 每槽每回合独立采样 `c ~ U[0.2,0.6]`，回合内固定；`vy=wz=0`；无指令速度课程。
- 回合上限：平地 1500 控制步（30 s）；楼梯 `ceil((2*(2.2 + N*sqrt(h*h+0.2*0.2))/c + 10)/0.02)`。参考：`(4,0.08,0.4)→1266`、`(0,0,0.4)→1500`、`(40,0.2,0.2)→7257`、`(40,0.2,0.6)→2753`。

### 5.4 PPO（与 r1 完全一致）

| 参数 | 值 |
|---|---|
| num_envs / rollout | 512 / 48 |
| epochs × minibatches | 5 × 4 = 20 Adam step/更新 |
| 每更新控制样本 / 物理子步 | 24576 / 98304（严格 4×） |
| initial lr / min / max | `5e-5` / `1e-5` / `1e-4` |
| desired KL / 自适应 | `0.01`；`kl>2×0.01` 则 lr/1.5，`0<kl<0.005` 则 lr×1.5，裁到 `[1e-5,1e-4]` |
| gamma / GAE λ | `0.995` / `0.95`（`gamma*lambda=0.94525`，半衰期约 2.766 s / 0.246 s） |
| clip / value coef / max grad norm | `0.2` / `1.0` / `1.0` |
| clipped value loss / entropy | `true` / `0.005` |
| 初始 std / std 范围 | 全部 `0.15` / `[0.10,0.30]` |
| 均值激活 | 无 tanh、无 clip |

### 5.5 奖励（`0.5-r2` 数值与 r1 逐字相同，仅身份字段改名）

`r_total = dt*(speed_rate + stall_rate + Σ other_weighted_rates) + footfall + swing + success + failure`，`dt=0.02` 只乘一次；事件不乘 dt；总奖励不 clamp、无 alive bonus；旧 `body_x_progress` 恒 0、仅诊断；`route_error` 不出现在任何项。

| 连续项 | 权重 |
|---|---:|
| track_angular_velocity（活动时减基线 1） | +1.0 |
| pose（活动时减基线 1；sigma `[0.15,0.35,0.5]×4`，零速 `[0.05,0.1,0.15]×4`） | +0.05 |
| orientation：`roll² + relu(pitch-55°)² + 4*relu(-5°-pitch)²` | -1.0 |
| body_ang_vel | -0.05 |
| angular_momentum | -0.025 |
| joint_acc | -2.5e-7 |
| joint_limits | -10.0 |
| action_rate | -0.02 |
| slip | -0.25 |
| soft_landing | -0.001 |
| stand_still | -1.0 |
| center `min((local_y/0.2)²,4)·I(active)` | -0.5 |
| heading `(1-cos(e_world))·I(active)` | -0.5 |
| edge（42 mm 内缩） | -1.0 |
| nonfoot（>10 N） | -1.0 |

事件：落脚 `0.50`/足/阶/回合；抬脚 `0.15`/足/目标阶（窗口 0.60 m）；成功 `+10`；真实失败 `-10`（同帧只结算失败）；普通 timeout 为 0（保留 bootstrap）。

速度项 `u=vx_b/c`、`ey=vy_b/0.2`、`a=1.5·exp(-4(u-1)²-ey²)-0.5`：

```text
over = 0.5*min(relu(u-1.2)^2, 4)
back = 0.5*min(relu(-u)^2, 4)
side = 0.25*min(ey^2, 4)
speed_rate = g*max(a,0) + min(a,0) - over - back - side
r_speed    = 0.02 * speed_rate
```

真值（`vy=0,g=1`，容差 1e-6）：`u=0 → -0.472527`、`0.25 → -0.341901`、`0.5 → 0.051819`、`1 → 1`、`1.2 → 0.778216`、`1.5 → 0.006819`、`2 → -0.792527`、`3 → -2.12`；停速 `-0.4725265417/s`（每步 `-0.009450530833`）。

停滞：窗口 51 个位置样本（50 间隔 = 1.00 s）+ 51 个高水位样本；`dx<0.15*c*1.00` 且 `dM<0.03`（严格小于）为 qualifying，连续 50 帧后 `stall_rate=-0.25`，每步 `0.02×(-0.25) = -0.005`（`-0.004999999888` 的 float32 表示）；最早 `t=99`；不新增停滞终止、不受 g 门控、成功帧关闭。

### 5.6 评估与门禁（r1 规则不变）

- 三速顺序 `0.2/0.4/0.6`，每档 20 回合；每档 `case_pass_count ≥ 18/20` 为一次完整通过，连续两次晋级；final test 每档 100 回合、`≥95/100`。
- `case_pass`：F0 = 完成 500 步且无 failure 且全程 `|y|≤0.20 m`、`|yaw|≤10°` 且 speed_pass；P/S = 安全到达且非 failure（`joint_success` 只作速度诊断）。
- 速度窗口：F0 step 101～500（400 采样）；P：首次 `x≥-0.5` 进入、首次 `x≥N*0.2+0.5` 结束；S：`x≥0 → x≥N*0.2`。平滑 MAE ≤0.30、平均相对误差 ≤0.20、最长偏离 ≤50 采样。
- case seed：`validation/flat_retention = 200300+seed`、`perturbed = 300300+seed`、`diagnostic = 500300+seed`、`final test = 400342`。
- 每 50 更新退化诊断（3 速 × 5 回合，固定 `[-1,0,0]`）；从阶段更新 100 起，15 例中 ≥12 例停滞占比 ≥0.80 且楼梯阶段同批 ≥12 例未获首阶 → `collapse_hit`，连续两次即 `behavior_collapse`。
- 资源守卫：整卡 ≤7168 MiB、进程 ≤5120 MiB、host available ≥4096 MiB、磁盘 free ≥10 GiB；超限停止，不自动减 `num_envs`。
- 每 100 更新常规保存；每 50 更新诊断及对应 checkpoint；阶段开始/切换/结束、受控暂停、退化终态额外保存。

### 5.7 checkpoint schema 7

顶层 14 键精确集合：`schema, identity, contract_sha256, model, optimizer, normalizer, learning, rng, curriculum, ledger, evaluation, source, recipe, resume_mode`。`resume_mode='new_episode'`。

必须拒绝：r1 完整 checkpoint（`critic.0.weight` 为 `[512,208]`、缺 `actor.route_fusion.*`）、`schema≤6`/无 schema 标记、`v05-smoke-state-v1` 与 `v05-r2-smoke-state-v1`、r1/r2 initialization 容器（`scope='initialization'`）、未知尾部键/部分 Adam。

## 6. 文件清单（0.5-r2）

| 路径 | 职责 | 归属 |
|---|---|---|
| `scripts/a2_v05_r2_bootstrap.py` | 仅 stdlib 的 `src.tasks` namespace 隔离 | A2 |
| `scripts/check_a2_ideal_stairs_v05_r2_migration.py` | CPU 迁移审计入口 | A2 |
| `src/tasks/velocity/v05_r2/{spec,model,rewards,env,speed,ppo,runner,recovery,evaluation,telemetry,connection}.py` | 生产模块 | 其他工作包 |
| `src/tasks/velocity/v05_r2/orchestration.py` | pilot/formal/probe 状态机、supervisor、2500/30000 同源预算 | **缺失**（见 §9.1） |
| `scripts/run_a2_ideal_stairs_v05_r2.py` | supervisor 薄入口（只 activate + 委托） | 本轮 |
| `scripts/train_a2_ideal_stairs_v05_r2.py` | 常驻 worker（supervisor 调用；只接线） | 本轮 |
| `scripts/evaluate_a2_ideal_stairs_v05_r2.py` | 冻结模型评估入口 | 本轮 |
| `scripts/play_a2_ideal_stairs_v05_r2.py` | 单速回放入口；三速分别成目录 | 本轮 |
| `scripts/smoke_a2_ideal_stairs_v05_r2.py` | 8 环境 × 2 更新工程冒烟（含 route 检查） | 本轮 |
| `scripts/test_a2_ideal_stairs_v05_r2.py` | pytest CPU 跑批入口（默认 `test_v05_r2_*.py`） | 本轮 |
| `scripts/diagnostics/analyze_a2_v05_r2_rewards.py` | W2 反例/锚点本地复验 + 旧表索引 | 本轮 |
| `tests/test_v05_r2_reward_analysis.py` | 分析脚本不变量测试 | 本轮 |

## 7. 真实命令模板（关键参数全部显式写值）

统一 Python：`/home/xxs/anaconda3/envs/a2/bin/python`；headless 设 `MUJOCO_GL=egl`；CPU 命令显式 `CUDA_VISIBLE_DEVICES=`。所有入口都在 `import src.tasks` 之前 activate `scripts/a2_v05_r2_bootstrap.py`。

### 7.1 CPU 测试（SDS §10.2）

> **必须走本入口**，不要直接 `pytest tests/test_v05_r2_*.py`：旧 `src/tasks/__init__.py` 的全量扫描会先执行，bootstrap 之后拒绝替换。

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
CUDA_VISIBLE_DEVICES= /home/xxs/anaconda3/envs/a2/bin/python \
  scripts/test_a2_ideal_stairs_v05_r2.py --pattern 'test_v05_r2_*.py'
```

可选：`--verbose`、`--log <新文件>`（不覆盖）、`--keyword/-k`、`--maxfail N`。单文件：

```bash
CUDA_VISIBLE_DEVICES= /home/xxs/anaconda3/envs/a2/bin/python \
  scripts/test_a2_ideal_stairs_v05_r2.py --pattern 'test_v05_r2_reward_analysis.py' --verbose
```

### 7.2 ETH 迁移审计（CPU；seed42/43 各一次，A2 包脚本）

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
CUDA_VISIBLE_DEVICES= /home/xxs/anaconda3/envs/a2/bin/python \
  scripts/check_a2_ideal_stairs_v05_r2_migration.py \
  --mode audit --device cpu --seed 42 \
  --project-root /home/xxs/dog/unitree_rl_mjlab_a2 \
  --source-checkpoint /home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt \
  --source-checkpoint-sha256 8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722 \
  --output-dir logs/a2_ideal_stairs_v05_r2/implementation/migration_seed42
```

seed43：同命令改 `--seed 43` 与 `--output-dir logs/a2_ideal_stairs_v05_r2/implementation/migration_seed43`。

### 7.3 W2 反例 / 锚点本地复验（本工作包）

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
CUDA_VISIBLE_DEVICES= /home/xxs/anaconda3/envs/a2/bin/python \
  scripts/diagnostics/analyze_a2_v05_r2_rewards.py \
  --output-dir logs/a2_ideal_stairs_v05_r2/implementation/analysis
```

输出 `a2_v05_r2_reward_analysis.json`、`a2_v05_r2_reward_analysis.md`、`analysis_index.json`；已存在时需 `--force` 才替换这三个 r2 输出（旧表永不改写）。

### 7.4 pilot 预检 / 执行（SDS §10.4）

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
/home/xxs/anaconda3/envs/a2/bin/python scripts/run_a2_ideal_stairs_v05_r2.py \
  --mode pilot \
  --task Unitree-A2-IdealHeightMap-Stairs-V05R2 \
  --recipe-version 0.5-r2 --sds-revision 1.0 \
  --seed 42 --device cuda:0 \
  --num-envs 512 --eval-num-envs 8 \
  --vx-min 0.2 --vx-max 0.6 --vy 0.0 --wz 0.0 \
  --save-interval 100 --diagnostic-interval 50 \
  --stage-budgets F0=2000,P1=500 \
  --pilot-p1-updates 500 \
  --init-checkpoint /home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt \
  --init-checkpoint-sha256 8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722 \
  --output-dir logs/a2_ideal_stairs_v05_r2/pilot_seed42_trial01
```

- 无 `--execute`＝只预检：只写 `logs/a2_ideal_stairs_v05_r2/implementation/preflight/`，**不建 run 目录、不启动 GPU、不联网写入**。获训练指令后同命令加 `--execute`。
- seed43：改 `--seed 43`、`--output-dir logs/a2_ideal_stairs_v05_r2/pilot_seed43_trial01`，其余配方完全相同。
- 预检必须断言 `total_budget=2500`、`update_cap=2500`、`F0=2000`、`P1=500`。
- **`--max-updates` 的归属（偏差，待裁决）**：SDS §10.4/§10.6 的模板把 `--max-updates 2500/30000` 写在 supervisor 命令行里，但冻结的 `v05_r2/orchestration.parser()` 没有这个参数，真正接受 `--max-updates` 的是 **worker**（`train_..._r2.py`，由 `Supervisor.worker_command()` 显式传入 `mode_budget(mode)['total']`）。因此本文的命令模板**不给 supervisor 传 `--max-updates`**（传了会被 argparse 拒绝），有效上限由 supervisor 从同一 `mode_budget` 计算并写入预检记录/结果；worker 端强制 `--max-updates == mode_budget(mode).total`。该差异已列入 §9.1 待裁决项。

### 7.5 formal 预检 / 执行（SDS §10.6；需两个真实通过的 pilot 报告）

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
/home/xxs/anaconda3/envs/a2/bin/python scripts/run_a2_ideal_stairs_v05_r2.py \
  --mode formal \
  --task Unitree-A2-IdealHeightMap-Stairs-V05R2 \
  --recipe-version 0.5-r2 --sds-revision 1.0 \
  --seed 42 --device cuda:0 \
  --num-envs 512 --eval-num-envs 8 \
  --vx-min 0.2 --vx-max 0.6 --vy 0.0 --wz 0.0 \
  --save-interval 100 --diagnostic-interval 50 \
  --stage-budgets F0=2000,P1=1000,P2=1000,P3=1000,P4=2000,S1=3000,S2=4000,S3=4000,S4=5000,S5=7000 \
  --init-checkpoint /home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt \
  --init-checkpoint-sha256 8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722 \
  --pilot-report-42 logs/a2_ideal_stairs_v05_r2/pilot_seed42_trial01/report.json \
  --pilot-report-43 logs/a2_ideal_stairs_v05_r2/pilot_seed43_trial01/report.json \
  --output-dir logs/a2_ideal_stairs_v05_r2/formal_seed42_trial01
```

- §10.6 是**预检模板**；满足准入后实际执行必须显式加 `--execute`，其余关键参数按 SDS 全量写出。**不得把成功预检写成已训练**。
- formal 去掉 `--pilot-p1-updates`；两个 report 必须是真实路径且通过 SHA 校验；预检必须断言 `total_budget=30000`、`update_cap=30000`。
- 源码或配方一旦变化，旧 pilot 自动失效，不得挑选性复用。

### 7.6 W7 资源 probe（`--mode probe`；训练指令后）

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
/home/xxs/anaconda3/envs/a2/bin/python scripts/run_a2_ideal_stairs_v05_r2.py \
  --mode probe \
  --task Unitree-A2-IdealHeightMap-Stairs-V05R2 \
  --recipe-version 0.5-r2 --sds-revision 1.0 \
  --seed 42 --device cuda:0 \
  --num-envs 512 --eval-num-envs 8 \
  --vx-min 0.2 --vx-max 0.6 --vy 0.0 --wz 0.0 \
  --save-interval 100 --diagnostic-interval 50 \
  --stage-budgets F0=2000,P1=1000,P2=1000,P3=1000,P4=2000,S1=3000,S2=4000,S3=4000,S4=5000,S5=7000 \
  --init-checkpoint /home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt \
  --init-checkpoint-sha256 8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722 \
  --output-dir logs/a2_ideal_stairs_v05_r2/w7_probe_trial01
```

probe 契约（由 `orchestration.validate_arguments` 断言，`run` 原样转发）：seed 42、**完整 formal 十阶段预算表**（只作配置身份）、禁止 `--pilot-p1-updates`/`--pilot-report-*`、禁止 `--resume-run`。实际学习工作量严格 **5+50+1=56** 更新（1,376,256 控制样本），评估/S5 三速无学习探针成本**另列**；**不得因 formal 表上限启动 30000 更新**。probe 与 smoke 不产生 pilot/formal 初始化来源。

### 7.7 evaluate（冻结模型，`--num-envs 8`）

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
MUJOCO_GL=egl /home/xxs/anaconda3/envs/a2/bin/python \
  scripts/evaluate_a2_ideal_stairs_v05_r2.py \
  --task Unitree-A2-IdealHeightMap-Stairs-V05R2 \
  --checkpoint <r2-checkpoint.pt> \
  --checkpoint-sha256 <该 checkpoint 的实测 64 位小写 sha256> \
  --stage S5 --kind validation --seed 42 --num-envs 8 --device cuda:0 \
  --mode formal --video \
  --output-dir logs/a2_ideal_stairs_v05_r2/eval/seed42_S5_validation_update<NNNNNN>
```

- 必填：`--checkpoint`、`--checkpoint-sha256`、`--stage`、`--kind`、`--seed`、`--num-envs 8`、`--output-dir`（必须不存在）。
- `kind ∈ {validation, perturbed, diagnostic, test, flat_retention}`；`test` 仅 S5、`perturbed` 仅 S1；`diagnostic` 15 例。
- `--stage` 必须等于 checkpoint 课程阶段；更早阶段需显式 `--allow-earlier-stage`（结果标 `non_final_earlier_stage`、`final_acceptance=false`）；更晚阶段直接拒绝。
- `--mode` 必须等于 checkpoint `source['mode']`（`preload_checkpoint` 校验）；`probe` 模式只能用于非门禁 kind。
- 入口先 CPU 校验 SHA / schema7 / `actor.route_fusion.*` / `critic.0.weight=[512,211]`，再仿真；原始 NPZ 必须含 `route_error[steps,3]`、`local_y`、`yaw`，缺失按"原始证据缺失"处理。
- 输出：`manifest.json`、`episodes.jsonl`、`steps_env0.jsonl`、`window_batch_*.npz`、`report.json`、本入口的 `evaluation_entry.json`（含 `route_contract`/`route_evidence`）。退出码 0 = 评估完成且证据从磁盘复算通过；门禁未通过是数据（`gate_pass=false`），不影响退出码。

### 7.8 play（单速回放；S5 三速分别成独立目录）

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
MUJOCO_GL=egl /home/xxs/anaconda3/envs/a2/bin/python \
  scripts/play_a2_ideal_stairs_v05_r2.py \
  --task Unitree-A2-IdealHeightMap-Stairs-V05R2 \
  --checkpoint <r2-checkpoint.pt> \
  --checkpoint-sha256 <该 checkpoint 的实测 64 位小写 sha256> \
  --stage S5 --kind test --seed 42 --num-envs 1 --device cuda:0 \
  --vx 0.2 --vy 0.0 --wz 0.0 \
  --x -1.0 --y 0.0 --yaw 0.0 \
  --max-steps 12000 --video --mode formal \
  --output-dir logs/a2_ideal_stairs_v05_r2/play/seed42_S5_vx0.2
```

三速必须三条独立命令、三个独立目录（不合并、不覆盖）；`--max-steps` 显式取该速协议回合上限：

```bash
CK=<S5 checkpoint 路径>; SHA=<对应 64 位 sha256>
for VX in 0.2 0.4 0.6; do
  case $VX in 0.2) MAX=7257;; 0.4) MAX=3879;; 0.6) MAX=2753;; esac
  MUJOCO_GL=egl /home/xxs/anaconda3/envs/a2/bin/python \
    scripts/play_a2_ideal_stairs_v05_r2.py \
    --task Unitree-A2-IdealHeightMap-Stairs-V05R2 \
    --checkpoint "$CK" --checkpoint-sha256 "$SHA" \
    --stage S5 --kind test --seed 42 --num-envs 1 --device cuda:0 --mode formal \
    --vx $VX --vy 0.0 --wz 0.0 --x -1.0 --y 0.0 --yaw 0.0 \
    --max-steps $MAX --video \
    --output-dir logs/a2_ideal_stairs_v05_r2/play/seed42_S5_vx${VX}
done
```

- 每速写 `<output-dir>/vx_<speed>/`：`summary.json`、`body_trace.npz`（含 `route_error`、`e_world`、`heading_deg`、`route_saturation`、`local_y`、`yaw`）、`vx_<speed>.mp4`；外加 `playback_index.json`。
- 回放**永不是门禁**：`gate_eligible=false`、`not_a_gate=true`、`final_acceptance=false`；提前阶段标 `non_final_earlier_stage`。
- `--max-steps` 缺省时：楼梯 = `spec.episode_step_limit(N,h,vx)`；flat（N=0）= 500 步观测窗口。显式值必须 ≥ 该最小值，避免裁剪 timeout/观测窗口。
- `route_error` 的维度/finite 会检查；非 finite 直接判失败（exit 1）。
- 设计限制：`v05_r2.evaluation` 没有 `replay` kind（每种 kind 回合数固定 20/100/15），所以回放走 `V05R2Env(evaluation=True, fixed_vx, initial_cases)` + `cpu_speed_oracle`，不伪造 case 表。

### 7.9 W6 GPU 工程 smoke（`--device cuda:0` 才是 GPU 验收）

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
MUJOCO_GL=egl /home/xxs/anaconda3/envs/a2/bin/python \
  scripts/smoke_a2_ideal_stairs_v05_r2.py \
  --task Unitree-A2-IdealHeightMap-Stairs-V05R2 \
  --recipe-version 0.5-r2 --sds-revision 1.0 \
  --seed 42 --device cuda:0 \
  --num-envs 8 --stage P1 --updates 2 \
  --rollout 48 --epochs 5 --minibatches 4 \
  --save-interval 100 --diagnostic-interval 50 \
  --init-checkpoint /home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt \
  --init-checkpoint-sha256 8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722 \
  --extra-steps 12 \
  --output-dir logs/a2_ideal_stairs_v05_r2/smoke_seed42_trial01
```

CPU dry run（仅工程验证，**不是** GPU 验收；状态固定 `passed_cpu_engineering_only`）：

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl /home/xxs/anaconda3/envs/a2/bin/python \
  scripts/smoke_a2_ideal_stairs_v05_r2.py \
  --task Unitree-A2-IdealHeightMap-Stairs-V05R2 \
  --recipe-version 0.5-r2 --sds-revision 1.0 \
  --seed 42 --device cpu \
  --num-envs 8 --stage P1 --updates 2 \
  --rollout 48 --epochs 5 --minibatches 4 \
  --save-interval 100 --diagnostic-interval 50 \
  --init-checkpoint /home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt \
  --init-checkpoint-sha256 8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722 \
  --extra-steps 12 \
  --output-dir logs/a2_ideal_stairs_v05_r2/implementation/smoke_cpu
```

自检项（任一失败 → `status=failed`、exit 1）：2 更新 / 40 Adam / 768 控制样本 / 3072 物理子步；奖励、观测、loss、梯度 finite；逐项加和误差 ≤1e-5；每次 Adam 后 std∈[0.10,0.30]；**五键观测含 `route_error`**（shape `[8,3]`、dtype float32、finite，storage 为 `[384,3]`，`route/nonfinite_count=0`）；迁移后 `route_fusion` 两个 tensor 逐元素 0、Critic `[512,211]`、`std=0.15`；schema7 `preload_checkpoint` 拒绝 `v05-r2-smoke-state-v1` 容器；额外无学习 F0/P1 混合环境固定动作检查（成本单列，不计入 768）。

### 7.10 worker 合同（supervisor 生成；不供人手调用）

```text
python scripts/train_a2_ideal_stairs_v05_r2.py \
  --task Unitree-A2-IdealHeightMap-Stairs-V05R2 --mode pilot|formal --seed 42|43 \
  --device cuda:0 --num-envs 512 --eval-num-envs 8 \
  --vx-min 0.2 --vx-max 0.6 --vy 0.0 --wz 0.0 \
  --save-interval 100 --diagnostic-interval 50 \
  --stage-budgets F0=2000,P1=500 [--pilot-p1-updates 500] \
  [--init-checkpoint <ETH> --init-checkpoint-sha256 <ETH_SHA> |
   --resume <checkpoint> --resume-sha256 <64hex>] \
  --output-dir <trial> --ledger-dir <trial>/ledger \
  --ledger-run-id v05-r2-{mode}-seed{seed} --control-file <supervisor control.json> \
  --max-updates 2500|30000
```

规则（实现见 `scripts/train_a2_ideal_stairs_v05_r2.py`）：

- `--init-checkpoint/--init-checkpoint-sha256` 与 `--resume/--resume-sha256` **互斥**，且必须恰好给出一组；`--init-checkpoint-sha256` 只能是冻结 ETH SHA。
- `--ledger-dir` 与 `--ledger-run-id` 必填；ledger 固定为 `<ledger-dir>/ledger.jsonl`，本次 trial 的每个 checkpoint 都由同一 ledger 提交。
- `--ledger-run-id` 必须等于 `v05-r2-{mode}-seed{seed}`（防止串 run）。
- `--max-updates` 必须等于 `mode_budget(mode).total`（2500 / 30000）；传其他值直接拒绝。
- resume 时输出目录必须已存在；`<ledger-dir>/ledger.jsonl` 不存在或 `V05Runner.resume` 抛 `RecoveryBlocked` → 写 `failure.json`（`status=recovery_blocked`）并非 0 退出，**不回退到 ETH 初始化**。
- worker 不含任何训练数学：PPO/课程/ledger/schema7/评估全部在 `v05_r2/*`。

## 8. 本轮已执行的验证与真实结果

### 8.1 入口 `--help`（CPU）

| 脚本 | 结果 |
|---|---|
| `test_a2_ideal_stairs_v05_r2.py` | exit 0；`--pattern` 默认 `test_v05_r2_*.py` |
| `train_a2_ideal_stairs_v05_r2.py` | exit 0；全部参数可见（含 `--ledger-dir/--ledger-run-id/--resume*/--max-updates`） |
| `evaluate_a2_ideal_stairs_v05_r2.py` | exit 0；`--num-envs {8}` 固定 |
| `play_a2_ideal_stairs_v05_r2.py` | exit 0；`--num-envs {1}`、`--vx {0.2,0.4,0.6}` |
| `smoke_a2_ideal_stairs_v05_r2.py` | exit 0；`--device {cuda:0,cpu}` |
| `run_a2_ideal_stairs_v05_r2.py` | exit 0；`--mode {pilot,formal,probe}`、`--execute`、`--resume-run` 可见，无 `--max-updates`（见 §7.4 偏差说明） |

### 8.2 预检与否定用例（真实执行）

真实命令见 §7.4/§7.6（`CUDA_VISIBLE_DEVICES=`，无 `--execute`）。实测：

| # | 命令形态 | 退出码 | 观察 |
|---|---|---:|---|
| 1 | `--mode pilot --seed 42` | 0 | 写 `preflight_pilot_seed42.json`；`status=preflight_validated`；`total_budget=2500=update_cap`；目标目录 `pilot_seed42_trial01` **未创建** |
| 2 | `--mode pilot --seed 43` | 0 | 同上（`preflight_pilot_seed43.json`，同一 `binding_sha256=0fef47ee…`，证明两 seed 合同相同） |
| 3 | `--mode probe --seed 42`（完整 formal 表） | 0 | 写 `preflight_probe_seed42.json`；`total_budget=update_cap=56`、`probe_learning_updates=56`；目标目录未创建 |
| 4 | `--mode probe --resume-run` | 2 | `W7 resource probe is a one-shot resource run; it is not resumable` |
| 5 | `--mode formal`（无 pilot 报告） | 2 | `formal training requires both pilot reports`（formal 预检必须等两个真实通过的 pilot） |
| 6 | `--mode pilot --stage-budgets F0=2000,P1=600` | 2 | `pilot stage budgets must be exactly F0=2000,P1=500` |

三条负向（4/5/6）都**没有**创建任何 run 目录。预检记录本身写明 `training_started=false`、`gpu_started=false`、`network_started=false`、`execute=false`、`effective_cap_source='mode_budget'`。

`run_a2_ideal_stairs_v05_r2.py` 只做两件事：activate bootstrap、把 `argv` 原样交给 `v05_r2.orchestration.main`；所有 CLI 规则、资源守卫、GPU 队列锁与 worker 启动都由该模块拥有，因此这里**没有**第二份校验。

### 8.3 CPU 测试（官方入口）

```bash
CUDA_VISIBLE_DEVICES= /home/xxs/anaconda3/envs/a2/bin/python \
  scripts/test_a2_ideal_stairs_v05_r2.py --pattern 'test_v05_r2_*.py'
```

实测（`tests/test_v05_r2_*.py`，含本工作包的 `test_v05_r2_reward_analysis.py`）：**331 collected / 331 passed / 0 failed / 0 skipped / 184 subtests / 345.2 s / exit 0**（另有 1 条 `torch.jit.script` 弃用警告，属上游）。

失例说明：本工作包早期版本曾在 `test_discount_conventions_and_additivity` 里用 `1e-9` 比较 **stall 折扣和**，而 r2 `StallState` 的每步金额是 float32（`-0.004999999888`），实测残差 `1.3528e-08`。按 r1 反例报告的口径改为 **`STALL_SUM_TOLERANCE=1e-7`**（并单独断言 float32 表示同一性到 `1e-12`），三锚点仍按 `1e-6`、float32 累加包络仍按 `1e-5`，未放宽其它断言。

### 8.4 分析脚本不变量测试

```bash
CUDA_VISIBLE_DEVICES= /home/xxs/anaconda3/envs/a2/bin/python \
  scripts/test_a2_ideal_stairs_v05_r2.py --pattern 'test_v05_r2_reward_analysis.py'
```

11 项不变量：七场景集合、三锚点、S3 rate/total、stall 首扣步、排序关系、缺失项声明、r2 SHA 绑定、旧表 superseded、折扣与可加性、route 只读、可复现性（两次运行同 digest、两个目录字节一致、重复写拒绝）。

### 8.5 W2 反例/锚点复验结果（r2 纯函数与状态机）

证据：`logs/a2_ideal_stairs_v05_r2/implementation/analysis/a2_v05_r2_reward_analysis.{json,md}` + `analysis_index.json`。

- **r2 源码闭包 SHA / r2 合同 SHA / 实施裁决 SHA** 都写入报告（`provenance`），并且 `source_sha = digest(排序后的 (path, sha256) 全闭包)`、`contract_sha = v05_r2.spec.contract_sha256()`；实测值见 §8.5b。闭包在同一次运行内被读两次并要求一致（`source_closure_stable=true`），生成期间的并发编辑会使该次运行直接失败而不是产出混合快照。
- 三个 r1 锚点在 **float64 参考口径**（同一 r2 函数、双精度输入）下复现：`3.9733841257`（1.5e-11）、`-2.4827493559`（1.4e-8）、`-6.8393842612`（1.7e-10）；**S3 rate `0.7382156834`（0 误差）与 S3 折扣和 `-0.6548597670`（3.3e-11）** 同时通过 1e-6。
- **float32（训练实际 dtype）** 同一批检查在声明包络 1e-5 内通过；S3 折扣和偏差 3.6e-6，来自 1000 步 float32 累加，不是公式差异。
- stall 首扣步：**S2=99、S7b=99、S6=384**（停车帧 288 不扣），由真实 `StallState`（51 样本 / 50 间隔 / 连续 50 帧）产生。
- 排序：匹配速度 > 静止；提前摔倒 < 静止；S3 侧移为负；S6 停车 < S1 直行；S5 纠偏 > S4 保持偏航。
- **明确声明**：本表只覆盖 speed/stall/center/heading/failure 五项、平地 N=0 合成序列，**不是完整训练回报**，不含 timeout/失败终止 bootstrap；局部合成收益不得冒充训练效果。
- 旧表处置：`q003_followup/r2_counterexamples.json`（`49e6c5e6…`）与 `r2_counterexamples.py`（`65290497…`）在 `analysis_index.json` 中标 **`superseded` / `current_evidence=false`**；`r2_counterexamples_v2.{py,json}` 标 `historical_reference_only`；r1 的 `reward_analysis.{json,md}` 标 `not_r2_evidence`。旧字节**未被覆盖**（脚本只读计算 SHA，索引记录 `bytes_unchanged=true`）。**r1 对拍结果不得直接作为 r2 验收**。

### 8.5b W2 证据的绑定值（生成当时的实测 SHA）

| 字段 | 值 |
|---|---|
| `r2_source_sha256`（整个 r2 源闭包，74 个文件） | `d2ba4a0f338622eef27cfceabe38979a772fc9a365b67c32c3d741a0ed21c2a9` |
| `r2_contract_sha256` | `ed2d34178b7a7b88537e49ab5556fdc96b5686a59819fc32ffd53da3c9a536e9` |
| `implementation_decision_sha256` | `0cd996e72c494a6cd981db067230710dbc8a5c0930d97bed1e86069a86895bf1` |
| `design_decision_sha256` | `78515bb951a96a8c69b9f10b90043a6312ed39124bc7898bd08b89e046d3a0fc` |
| `sds_sha256` | `9ff356b022e6205f0a181ff4048ef0f6e773742250d3174bf9b087986c8a4abc` |
| 报告 `digest` | `9bafeeb29eb1bd5aca971eb7c6e27d45810a5ce69956b606b13eca087d18a501` |
| 预检 `binding_sha256`（pilot42/43、probe） | `0fef47ee7e74…`（三条预检记录与当前树一致） |
| `source_closure_stable` | `true`（同一次运行内两次闭包扫描一致，说明生成期间没有并发编辑） |

（`r2_source_sha256`/`r2_contract_sha256`/`digest` 在上述任何 r2 文件改动后都会变化，必须重跑；见 §0。生成后 `v05_r2/orchestration.py` 仍在被 C2 编辑一次，本表已按最新树重新生成；**冻结后必须再跑一次**：
`CUDA_VISIBLE_DEVICES= python scripts/diagnostics/analyze_a2_v05_r2_rewards.py --output-dir logs/a2_ideal_stairs_v05_r2/implementation/analysis --force`。）

### 8.6 ETH 迁移审计（他人执行，本轮只读取）

| seed | 状态 | zero SHA256（前 8） | 绑定的源文件数 | 备注 |
|---|---|---:|---:|---|
| 42 | `passed` | `a010ee23` | 73 | `actor mean max_abs_error=0.0`、`route_zero_invariance` 通过 |
| 43 | `passed` | `21949af3` | 73 | 同上 |

该审计绑定生成当时的 r2 闭包 `contract_sha256`；任何后续源码编辑都会使它失效，必须在冻结后重跑。

### 8.7 worker CLI 接线（`Supervisor.worker_command()` → `train`）

用真实 `v05_r2.orchestration.Supervisor.worker_command()` 生成四条命令，逐条喂给
`scripts/train_a2_ideal_stairs_v05_r2.py` 的 parser + `validate`：
证据 `logs/a2_ideal_stairs_v05_r2/implementation/wiring_checks/worker_cli_validation.json`。

| 命令 | 结果 | `--max-updates` | `--ledger-run-id` |
|---|---|---:|---|
| pilot seed42 | 接受 | 2500 = `mode_budget('pilot').total` | `v05-r2-pilot-seed42` |
| pilot seed43 | 接受 | 2500 | `v05-r2-pilot-seed43` |
| formal seed42（init） | 接受 | 30000 = `mode_budget('formal').total` | `v05-r2-formal-seed42` |
| formal seed42（resume） | 接受 | 30000 | `v05-r2-formal-seed42` |

6 条否定用例全部 exit 2：`resume + init` 同给、resume 缺 `--resume-sha256`、缺 `--ledger-dir`、
缺 `--ledger-run-id`、`--ledger-run-id` 与 mode/seed 不符、`--max-updates 29500`（r1 的十阶段求和）。
即：worker 侧强制"`--max-updates` 必须等于该 mode 的有效上限"，与 supervisor 生成的值逐条一致。

### 8.8 evaluate / play 的 fail-closed 负向检查

本轮**没有**真实 schema7 checkpoint（不伪造），因此只执行 fail-closed 路径（`--device cpu` / 不进入仿真）：

| 入口 | 输入 | 结果 |
|---|---|---|
| evaluate | `--num-envs 4` | argparse 拒绝（`choose from 8`），exit 2 |
| evaluate | 64 位但错误的 `--checkpoint-sha256` | `ValueError: Checkpoint SHA mismatch`（仿真前） |
| evaluate | `migration_zero.pt`（initialization 容器）+ 正确 SHA | `ForeignContainerError: initialization container (scope=initialization) is migration evidence, not a resume checkpoint`（SDS §7.5-4） |
| play | `--max-steps 200`（F0 观测窗口 500） | 拒绝：`--max-steps 200 is shorter than the protocol minimum 500 ... (episode cap 1500, F0 observation window 500)` |
| play | `--num-envs 2` | argparse 拒绝（`choose from 1`） |
| play | 已存在的 `--output-dir` | `--output-dir must not exist` |

两条 preload 拒绝以未捕获异常形式退出（exit 1，打印 traceback），与 r1 入口一致；这是"fail-closed 优先"的选择，不是崩溃。

### 8.9 CPU 工程 smoke（`--device cpu`，**不是** GPU 验收）

命令见 §7.9 的 CPU dry run；证据 `logs/a2_ideal_stairs_v05_r2/implementation/smoke_cpu/smoke.json`。

| 项 | 实测 |
|---|---|
| `status` / `exit_code` | `passed_cpu_engineering_only` / `0`（`failures=[]`） |
| Adam step | 40/40（2 更新 × 5 epochs × 4 minibatches） |
| 控制样本 / 物理子步 | 768/768 / 3072 |
| 逐项加和误差 | `0`（容差 1e-5） |
| std 范围（40 次 Adam 后） | `[0.14953966, 0.15037753]` ⊂ `[0.10,0.30]` |
| 迁移不变量 | `route_fusion.weight [512,3]` 与 `.bias [512]` 逐元素 0；`critic.0.weight [512,211]`；`critic_normalizer` 80 维；`std=0.15` |
| 五键观测 / route | 96 次逐步检查（2×48 控制步）、`shape_errors=[]`、`nonfinite_count=0`、storage `[384,3]`×2、`normalized_abs_y_max=0.2907`、`heading_deg_max=4.88°` |
| schema7 预加载拒绝 | `rejected=true`（容器 `v05-r2-smoke-state-v1`、`formal_resumable=false`） |
| 无学习混合检查（成本单列，不计入 768） | P1：`tiles=[0,1]`、96 样本、`reward_sum_ok=true`、`route_ok=true`；F0：`tiles=[0]`、96 样本、同样通过 |
| 资源 | `elapsed≈11.7 s`、`peak_rss≈2.04 GiB`（CPU） |

该结果只证明**接线与自检**正确，不构成 GPU 验收，更不构成训练效果。

## 9. 已知限制与阻塞

### 9.1 已解除：`v05_r2/orchestration.py` 由 C2 落地（附一条待裁决偏差）

- SDS §8.1 把 pilot/formal/probe 状态机、supervisor、退化保护、2500/30000 同源预算、资源守卫、worker 子进程启动、pilot 报告绑定全部划给 `src/tasks/velocity/v05_r2/orchestration.py`。该文件现已由 C2 提供（`Supervisor` / `PilotSequence` / `run_resource_probe` / `parser` / `validate_arguments` / `main`）。
- 本工作包的 `run` 入口按**真实签名**接线：`from ...v05_r2.orchestration import main` → `main(argv)`；不复制任何校验。`train` worker 与它逐字段对齐：`ledger_run_id(mode, seed) = 'v05-r2-{mode}-seed{seed}'`（与 worker 的强制值相同）、`--max-updates = mode_budget(mode)['total']`、`--ledger-dir = <trial>/ledger`、pilot 追加 `--pilot-p1-updates 500`、resume 用 `--resume/--resume-sha256`（§8.7 已逐条验证）。
- **待裁决偏差**：SDS §10.4/§10.6 的 supervisor 命令模板包含 `--max-updates 2500/30000`，但冻结的 `orchestration.parser()` 不定义该参数（argparse 会拒绝），它只由 supervisor 传给 worker。二选一：①修订 SDS 模板（把 `--max-updates` 从 supervisor 命令行移除，改述为"由 supervisor 传入 worker"）；②由 C2 给 supervisor 增加该参数并断言等于 `mode_budget`。本工作包按①记录命令（见 §7.4），不擅自改 SDS 或他人模块。
- 仍需注意：`--mode probe` 的**预检形态**已实测只写 `implementation/preflight/`；`--resume-run` 对 probe 已被 `validate_arguments` 拒绝（§8.2 第 4 行）。

### 9.2 Q002 例外范围（不能被扩大解释）

- 状态三元并存：`accepted_with_known_limitation_Q002`、`long_horizon_numeric_equivalence=not_established`、`failed_q001`。
- Q002 只解除"长期轨迹数值等价"作为 W8/W9 训练准入的硬条件；Q002 §4.2 的四项工程证据仍必须保留。
- **该例外不覆盖** r2 新增的 `route_error` 观测、3→512 旁路、208→211 Critic、flat 初态随机化：这些必须按 SDS §2.6/§9.3 独立审计。也不得引用 Q002 历史证据作为 r2 的迁移/等价证据。
- 不得写"r2 与 r1 物理轨迹逐位一致"或"长期等价已证明"。

### 9.3 r1 证据不可复用

- r1 的 SDS、源码、日志、模型、pilot/formal 结果、case 表、视频、checkpoint SHA 均属 r1 合同；r2 架构（schema7/211/旁路）与 r1（schema6/208）不兼容，**均不得复用**为 r2 的训练、评估、迁移或门禁证据。
- r1 `total_budget=29500` 只作历史记录；r2 使用 2500/30000 单一来源。
- 禁止"从 r1 模型续训/挑旧 checkpoint"；pilot 与 formal 必须从 ETH 重新初始化。

### 9.4 `yaw_axis=0` 前提

- 当前 atlas 各 tile 只平移，`atlas_metadata()` 没有旋转字段，所以 `e_world = wrap_to_pi(yaw_world - 0)`，数值上等于 `local_yaw`。
- 该前提在 `spec`/`env` 中以显式常量与断言写明；**遇到需要支持旋转 tile/非零走廊轴的实例必须升级 `route_state` 合同（含部署语义）并向 Codex 提问**，不得静默按 0 计算，也不得自行扩展坐标。

### 9.5 部署（另议）

- `ideal_route_state` 的真实产生方式（传感器组合、标定、失效降级、硬件验证）属 SDS §13 未决问题，本版**不实现猜测方案**、**不声称部署就绪**。
- `route/*` 是否新增独立 telemetry panel（第 9 组）也属未决问题；本版按 `panel_count=8`、归入 `运动/接触`。

### 9.6 其他限制

- 选择路线②是针对信息缺口的干预，**不是已证明有效**；pilot 可能仍失败，失败须照实报告。
- `center`/`heading` 权重仍 `-0.5`、仍饱和；本版不解决"惩罚不足"，也不把 `-2.0` 作为 r2 方案。
- 平地方向信息仍依赖 `route_error` 与航向观测，不保证所有初态可恢复。
- 资源/时间估计依赖本机实测（约 12.6 min/100 更新，pilot 约 5.25 h/seed、formal 约 63 h）；外部 GPU 环境变化无法保证。
- 源码闭包随并行编辑失效（见 §0）：冻结前的任何证据都必须重跑。

## 10. 不得声称的内容

- 不得声称爬楼成功率、速度达标、停车/航向已改善，或"已解决停车"；这些只能由真实 pilot/formal/final test 的证据给出。
- 不得把 W6 工程通过写成 W8/W10 效果通过；不得把预检写成已训练；不得把 CPU dry run 写成 GPU 验收。
- 不得声称部署就绪或"部署依赖不变"：r2 新增 `ideal_route_state` 依赖且无硬件验证。
- 不得用本文件的 W2 局部合成表替代完整奖励验收（它只覆盖 5 项公式、平地 N=0）。
- 不得复用 r1 的任何 checkpoint/证据/视频充当 r2 结果。

## 11. 复现与环境

- 仓库：`/home/xxs/dog/unitree_rl_mjlab_a2`（工作区含大量未提交改动；本轮不 commit、不清理、不覆盖旧文件）。
- Python：`/home/xxs/anaconda3/envs/a2/bin/python`（3.11.18，torch 2.15.0+cu132）。
- headless：`MUJOCO_GL=egl`；CPU 命令显式 `CUDA_VISIBLE_DEVICES=`；不要在 `/tmp` 下当工作目录运行脚本（`/tmp/profile.py` 会遮蔽标准库）。
- 证据根：`logs/a2_ideal_stairs_v05_r2/implementation/{preflight,analysis,wiring_checks,smoke_cpu,migration_seed42,migration_seed43}/`。
- 关键证据文件：`preflight/preflight_pilot_seed42.json`、`preflight_pilot_seed43.json`、`preflight_probe_seed42.json`、`analysis/a2_v05_r2_reward_analysis.{json,md}`、`analysis/analysis_index.json`、`wiring_checks/worker_cli_validation.json`、`smoke_cpu/smoke.json`。
