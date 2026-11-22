# A2 理想高程图镂空楼梯 v0.5 实施与运行文档

- 代码版本：`0.5.0`；奖励配方：`0.5-r1`；checkpoint schema：`6`；SDS：`doc/sds/SDS_A2_IdealHeightMap_Stairs40_v0.5.md`（修订 1.1）。
- 正式楼梯协议：`v05-stair-climb-first-v1`；F0 平地协议：`v05-flat-speed-v1`。
- 任务身份：`Unitree-A2-IdealHeightMap-Stairs-V05`；日志根：`logs/a2_ideal_stairs_v05/`。
- 本文是运行文档，不冒充“已完成训练报告”。**训练效果只能引用真实测试结果**；W7～W10 已获用户授权，但必须在本文件“工程验收”完成后按 SDS 顺序执行，本文不预填任何未执行结果。

本文覆盖 D 工作包（入口脚本、诊断分析脚本、运行文档、分析测试）：
`scripts/{train,run,evaluate,play,smoke,test}_a2_ideal_stairs_v05.py`、
`scripts/diagnostics/analyze_a2_v05_rewards.py`、`tests/test_v05_reward_analysis.py`、本文件。
其余 v0.5 模块由 A/B/C 工作包拥有，本文只记录其真实接口与固定数值。

## 0. 已执行 / 未执行（一次说清）

| 项目 | 状态 | 证据 |
|---|---|---|
| ETH 迁移审计 seed42 | 已执行，`status=passed`（`max_abs_error=0.0`、`cuda_initialized=false`） | `implementation/migration_seed42_final/`（`migration_zero.pt` zero `d08b6d84…`，51 tensor 行，74 绑定源文件） |
| ETH 迁移审计 seed43 | 已执行，`status=passed`（`max_abs_error=0.0`、`cuda_initialized=false`） | `implementation/migration_seed43_final/`（zero `8f88c2e8…`） |
| CPU 测试 `--pattern 'test_v05_*.py'`（官方入口） | 已执行：**394 收集 / 373 通过 / 0 失败 / 21 跳过 / 179 subtests / 184.1 s，退出码 0**（§6.2；21 skipped = 20 GPU 环境用例 + 1 GPU 真实环境 e2e 用例） | `implementation/cpu_tests/final_frozen.log` |
| 诊断分析（反例/时间伸缩/trace） | 已执行 | `logs/a2_ideal_stairs_v05/implementation/analysis/reward_analysis.{json,md}` |
| pilot42 预检（无 `--execute`） | 已执行，exit 0、未建 run 目录（`source_sha=b30cf492…`、`binding_sha256=6ba289f2…`） | `implementation/preflight/preflight_pilot_seed42.json` |
| pilot43 预检（无 `--execute`） | 已执行，exit 0、未建 run 目录（`source_sha=b30cf492…`、`binding_sha256=6ba289f2…`） | `implementation/preflight/preflight_pilot_seed43.json` |
| W6 GPU smoke（`--device cuda:0`） | 已执行：`status=passed`、40/40 Adam、768/768 控制样本、3072 物理子步、逐项加和误差 0、std∈[0.149869,0.150424]、schema6 拒绝 smoke 容器 | `implementation/smoke_w6_gpu/smoke.json`（另有验收人两次独立复跑：`acceptance/smoke_gpu_audit{,_final}/`） |
| CPU smoke（`--device cpu`，工程 dry run） | 已执行，`passed_cpu_engineering_only` | `logs/a2_ideal_stairs_v05/implementation/smoke_cpu/smoke.json` + `smoke_cpu_stdout.txt` |
| GPU 环境测试 + 真实 F0 diagnostic e2e（cuda:0） | **已由独立验收人执行**：**28 passed / 23 subtests / 265.77 s**（27 个环境用例 + 真实 `V05Env`×`V05ActorCritic` 的 15 例 F0 诊断端到端） | `acceptance/Q002_acceptance_addendum.md` §5 第 2 行 |
| W7 资源 probe（首次真实执行） | **未通过**：55 更新（5 warmup + 50 measured）成功、四项资源门槛全过（整卡峰值 5121.25/7168 MiB、进程 3188/5120 MiB、host 10245.1/4096 MiB、磁盘 469.0/10 GiB、中位更新 5.65 s），但**共存评估阶段失败**（`evaluation mode must be pilot or formal`）；recovery 与 S5 段未执行。修复已落地（`evaluation.py` 支持 `mode='probe'`、`orchestration.py` 以 `mode='probe'` 提交），**重跑未执行** | `logs/a2_ideal_stairs_v05/W7_probe_seed42_20261115_183947/`（`probe/probe.json`、`final.json`） |
| run 预检 / `--resume-run` 反例 | 已由独立验收人执行：预检 `preflight_validated`、exit 0、目标目录未创建；三条 `--resume-run` 反例均 exit 2（互斥 / 缺 `run_contract.json` / 预算不符） | `implementation/acceptance/ACCEPTANCE_v05.md` §3 第 10–13 行 |
| 独立验收（工程侧） | 已完成并追加 Q002 补充签署：**阻断级偏差 0 项**；非阻断 7 项中 D-1/D-3/D-4/D-7 已修复/勘误，剩余见 §7.1；Q001/Q002 物理等价项不再阻塞 W8/W9；训练效果未验证 | `implementation/acceptance/ACCEPTANCE_v05.md`、`Q002_acceptance_addendum.md`（§8/§8.1 有摘要） |
| evaluate 真实仿真 | 已执行（CPU，临时 0 更新 schema6 checkpoint）：F0/validation 20 例/速 9 批、`evidence_verified=True`、`gate_pass=False`（0/20、10/20、11/20）；F0/diagnostic 15 例 `per_case` 齐全且 `diagnostic_decision` 可判（见 §6.6） | 接线验证目录 `implementation/wiring_checks/`（接线验证记录，不是门禁证据） |
| play 真实回放（F0/vx0.2，CPU） | 已执行，exit 0；`samples=400`、`speed_valid=True`、`speed_pass=False`、`gate_eligible=false`、`final_acceptance=false` | `implementation/wiring_checks/play_f0_vx02/`（接线验证用，不作为交付证据） |
| W8 两个 pilot / W9 formal / W10 final test | **已获授权；工程验收与 Q002 签署已完成；尚未执行**（W7 需先通过） | 授权：`implementation/coordination/training_authorization_receipt.json`；验收：`implementation/acceptance/ACCEPTANCE_v05.md` + `Q002_acceptance_addendum.md`；W7 见上一行 |

**源码闭包会随并行编辑失效**：schema-6 `contract_sha256` 绑定全部 v0.5 源文件与入口脚本（SDS §9.3-21）。并行工作包每次改动 `src/tasks/velocity/v05/*.py` 或本文件列出的入口脚本，都会使此前生成的 checkpoint 与证据在 `preload_checkpoint` 处被判为“contract/source closure differs”。因此**所有 W6 之后的证据必须在源码冻结后重新生成**，不能沿用编辑期间产生的 checkpoint。

## 1. 版本身份与冻结参数

| 项 | 值 |
|---|---|
| `CODE_VERSION` / `RECIPE_VERSION` / `SDS_REVISION` | `0.5.0` / `0.5-r1` / `1.1` |
| `SCHEMA_VERSION` | `6` |
| `TASK` / `PROTOCOL_ID` / `FLAT_PROTOCOL_ID` | 见文首 |
| `LOG_ROOT` | `logs/a2_ideal_stairs_v05/` |
| 唯一初始化权重 | `/home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt` |
| 初始化 SHA256 | `8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722` |
| 奖励配置 SHA256（当前源码） | `12cb9eb20144b776fea2cda8379d86098b11e9a0f3a5adf56915eb39be9bfed0`（由 `v05.spec.reward_config_sha256()` 计算；源码改动会改变此值） |
| 控制 | `physics dt=0.005 s`，`decimation=4`，`control dt=0.02 s` |
| 观测 | `proprio[45]`、`history[5,45]`、`heightmap[3,48,32]`、`privileged[35]`（不变项） |
| 几何 | 阶深 0.20 m、宽 0.80 m、踏板厚 0.01 m、平台长 2 m，无立板/侧板 |
| 网络 | 现有 `IdealHeightmapActorCritic`（结构、宽度、激活、融合不变） |
| 随机化 | 不新增推力/动力学/观测噪声；只保留本文初态分布 |

## 2. 文件清单与职责

| 路径 | 职责 |
|---|---|
| `scripts/a2_v05_bootstrap.py` | 仅 stdlib 的 `src.tasks` namespace 隔离（A 包）；所有入口的第一个 task 相关 import（幂等） |
| `src/tasks/velocity/v05/spec.py` | 单一身份/阶段表/槽位/奖励配方/PPO 配置/合同与源码闭包（A 包） |
| `src/tasks/velocity/v05/model.py` | `V05ActorCritic`、ETH 迁移、schema6 模型校验（A 包） |
| `src/tasks/velocity/v05/rewards.py` | 速度/停滞/摆动/落脚/连续项纯函数与状态机（A 包） |
| `src/tasks/velocity/v05/env.py` | `V05Env`：F0/P/S 真实 tile、混合槽位、奖励一次累计、pre-reset 证据（B 包） |
| `src/tasks/velocity/v05/speed.py` | F0/P 独立窗口、S 原 oracle、CPU float64 复算（B 包） |
| `src/tasks/velocity/v05/evaluation.py` | 固定 case 表、原始 NPZ/JSONL、`18/20`、`95/100` 门禁（B 包） |
| `src/tasks/velocity/v05/ppo.py` | PPO 数学只读测量适配 + 显式 `PPOConfig`（C 包） |
| `src/tasks/velocity/v05/runner.py` | 常驻训练循环、阶段预算、诊断/门禁边界、schema6 保存/恢复（C 包） |
| `src/tasks/velocity/v05/recovery.py` | schema6/ledger/评估事务/RNG/未知尾部（C 包） |
| `src/tasks/velocity/v05/orchestration.py` | pilot/formal 状态机、报告绑定、资源守卫、worker 子进程（C 包） |
| `src/tasks/velocity/v05/telemetry.py` / `connection.py` | 0.5 命名、8 面板、可靠远端投递（C 包） |
| `scripts/check_a2_ideal_stairs_v05_migration.py` | CPU 迁移审计入口（A 包，见 §5.2） |
| `scripts/run_a2_ideal_stairs_v05.py` | supervisor 入口（D 包）：pilot/formal 预检、执行与 resume 委托 |
| `scripts/train_a2_ideal_stairs_v05.py` | 常驻 worker（D 包）：由 supervisor 调用；`--ledger-dir/--ledger-run-id` 拥有 durable ledger，`--resume/--resume-sha256` 与 init 互斥、恢复失败 fail-closed |
| `scripts/evaluate_a2_ideal_stairs_v05.py` | 冻结模型评估入口（D 包） |
| `scripts/play_a2_ideal_stairs_v05.py` | 单速回放入口（D 包），S5 三速分别成目录；回合上限取 `spec.episode_step_limit`，flat 回放用 500 步观测窗口 |
| `scripts/smoke_a2_ideal_stairs_v05.py` | 8 环境×2 更新工程冒烟（D 包，独立于 runner/ledger） |
| `scripts/test_a2_ideal_stairs_v05.py` | pytest CPU 测试跑批入口（D 包） |
| `scripts/diagnostics/analyze_a2_v05_rewards.py` | 反例/时间伸缩/trace 重算分析（D 包） |
| `tests/test_v05_reward_analysis.py` | 分析脚本不变量测试（D 包） |

## 3. 完整数值配置

### 3.1 阶段表（tile/N/h/预算/门禁间隔）

| stage | tile_id | N | h(m) | 更新上限 | 完整门禁间隔 |
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

总预算 30000 更新 = 737,280,000 控制样本 = 600,000 Adam step。pilot 预算：`F0=2000,P1=500`（P1 固定 500，不提前晋级）。上表与 `spec.STAGES` 逐项一致。

### 3.2 槽位（固定，不允许 allocator 追加）

| stage | 512 环境 `flat/previous/current` | 8 环境 `flat/previous/current` | 8 环境 tile |
|---|---|---|---|
| F0 | 512 / 0 / 0 | 8 / 0 / 0 | `[0×8]` |
| P1 | 102 / 0 / 410 | 2 / 0 / 6 | `[0,0,1,1,1,1,1,1]` |
| P2 | 51 / 102 / 359 | 1 / 1 / 6 | `[0,1,2,2,2,2,2,2]` |
| P3 | 51 / 102 / 359 | 1 / 1 / 6 | `[0,2,3,3,3,3,3,3]` |
| P4 | 51 / 102 / 359 | 1 / 1 / 6 | `[0,3,4,4,4,4,4,4]` |
| S1 | 51 / 102 / 359 | 1 / 1 / 6 | `[0,4,5,5,5,5,5,5]` |
| S2 | 51 / 102 / 359 | 1 / 1 / 6 | `[0,5,6,6,6,6,6,6]` |
| S3 | 51 / 102 / 359 | 1 / 1 / 6 | `[0,6,7,7,7,7,7,7]` |
| S4 | 51 / 102 / 359 | 1 / 1 / 6 | `[0,7,8,8,8,8,8,8]` |
| S5 | 51 / 102 / 359 | 1 / 1 / 6 | `[0,8,9,9,9,9,9,9]` |

- `previous` 槽使用**紧邻前一阶段自己的 N/h**（例如 S1 的 previous = P4），不是 supervisor 当前阶段。
- 每槽每回合独立采样 `c ~ U[0.2,0.6]`，回合内固定；`vy=wz=0`。
- 平地回合上限 1500 控制步（30 s）；楼梯上限 `ceil((2*(2.2+N*sqrt(h*h+0.04))/c+10)/0.02)`，按该槽真实 N/h/c 计算（N=4,h=0.08,c=0.4 → 1266 步）。

### 3.3 PPO（`v05.ppo.PPOConfig`，冻结）

| 参数 | 值 |
|---|---|
| num_envs / rollout | 512 / 48 |
| epochs × minibatches | 5 × 4 = 20 Adam step/更新 |
| 每更新控制样本 / 物理子步 | 24576 / 98304 |
| initial lr / min / max | `5e-5` / `1e-5` / `1e-4` |
| desired KL / 自适应 | `0.01`；`kl>2×0.01` 则 lr/1.5，`0<kl<0.005` 则 lr×1.5，裁到 [1e-5,1e-4] |
| gamma / GAE λ | `0.995` / `0.95` |
| clip / value coef / max grad norm | `0.2` / `1.0` / `1.0` |
| clipped value loss / entropy | `true` / `0.005` |
| 初始 std / std 范围 | 全部 `0.15` / `[0.10,0.30]`（`action_distribution` 与 `constrain_std` 两处、每次 Adam 后投影） |
| 均值激活 | 无 tanh、无 clip |

### 3.4 奖励总式与权重

`r_total = dt*(speed_rate + stall_rate + Σ other_weighted_rates) + footfall + swing + success + failure`，`dt=0.02` 只乘一次；事件不乘 dt；总奖励不 clamp、无 alive bonus；旧 `body_x_progress` 恒 0、仅诊断。

| 连续项 | 权重 |
|---|---:|
| track_angular_velocity（活动时减基线 1） | +1.0 |
| pose（活动时减基线 1；sigma `[0.15,0.35,0.5]×4`，零速 `[0.05,0.1,0.15]×4`） | +0.05 |
| orientation：`roll² + relu(pitch-55°)² + 4*relu(-5°-pitch)²` | -1.0 |
| body_ang_vel（世界 roll/pitch 角速度平方和） | -0.05 |
| angular_momentum | -0.025 |
| joint_acc | -2.5e-7 |
| joint_limits | -10.0 |
| action_rate | -0.02 |
| slip（实际接触脚水平世界速度平方和，active） | -0.25 |
| soft_landing（接触时长 ≤ dt+1e-7 的足力范数和，active） | -0.001 |
| stand_still（不是本版 stall） | -1.0 |
| center `min((local_y/0.2)²,4)·I(active)` | -0.5 |
| heading `(1-cos(local_yaw))·I(active)` | -0.5 |
| edge（42 mm 内缩距离，四足平均） | -1.0 |
| nonfoot（任一监控连杆力范数 >10 N，不按连杆平均） | -1.0 |

事件金额：落脚 `0.50`/足/阶/回合；抬脚 `0.15`/足/目标阶（窗口 0.60 m）；成功 `+10`；真实失败 `-10`（同帧只结算失败）；普通 timeout 事件 0（保留时间截断 bootstrap）。

### 3.5 速度项（0.5 新公式）

`u=vx_b/c`，`ey=vy_b/0.2`，`a=1.5·exp(-4(u-1)²-ey²)-0.5`：

```
over = 0.5*min(relu(u-1.2)^2, 4)
back = 0.5*min(relu(-u)^2, 4)
side = 0.25*min(ey^2, 4)
speed_rate = g*max(a,0) + min(a,0) - over - back - side
r_speed = 0.02 * speed_rate
```

每秒率真值（`vy=0,g=1`，容差 1e-6）：

| u | 0 | 0.25 | 0.5 | 1 | 1.2 | 1.5 | 2 | 3 |
|---|---|---|---|---|---|---|---|---|
| rate | -0.472527 | -0.341901 | 0.051819 | 1 | 0.778216 | 0.006819 | -0.792527 | -2.120000 |

停止时约 `-0.472527/s`（每步 -0.00945053）；匹配指令 `+1/s`（每步 0.02）；超速死区 1.2c。门控 `g` 只关闭正部；负部、停滞与所有正则不受门控。

### 3.6 停滞（StallState）

窗口 51 个位置样本（50 个控制间隔 = 1.00 s）+ 51 个历史最大位置；`dx<0.15*c*1.0` 且 `dM<0.03`（严格小于）为 qualifying；连续 50 个 qualifying 帧（`t=99` 首次）后 `stall_rate=-0.25`，每步 `0.02×(-0.25) = -0.005`；不新增停滞终止、不受 g 门控、成功帧关闭。

### 3.7 评估与门禁阈值

- 三速顺序 `0.2/0.4/0.6`，每档 20 回合；每档 `case_pass_count ≥ 18/20` 为一次完整通过，连续两次晋级；final test 每档 100 回合、`≥ 95/100`。
- `case_pass`：F0 = 完成 500 控制步且无 failure 且全程 `|y|≤0.20 m`、`|yaw|≤10°` 且 speed_pass；P/S = 安全到达且非 failure（`joint_success` 只作速度诊断）。
- 速度窗口：F0 step 101～500（400 采样）；P：首次 `x≥-0.5` 进入、首次 `x≥N*0.2+0.5` 结束；S：原 oracle `x≥0 → x≥N*0.2`。平滑 MAE ≤0.30、平均相对误差 ≤0.20、最长偏离 ≤50 采样、`>0.50+1e-6` 判偏离。
- case seed：validation/flat_retention `200300+seed`、perturbed `300300+seed`、diagnostic `500300+seed`、final test `400342`。
- 每 50 更新退化诊断（3 速×5 回合，固定 `[-1,0,0]`）；阶段更新 100 起、15 例中 ≥12 例停滞占比 ≥0.80 且楼梯阶段同批 ≥12 例未获首阶 → collapse_hit，连续两次即 behavior_collapse。
- 回合/观察上限在设计上的**唯一来源**是 `v05.spec`：`spec.episode_step_limit(tile_n, tile_h, c)`（flat `N=0` 或 `h=0` 返回平地回合上限 1500；楼梯返回 `spec.stair_episode_steps`）与 `spec.F0_OBSERVE_STEPS=500`（F0 观测窗口）。S1 在 0.2/0.4/0.6 的回合上限分别为 2031/1266/1011；S5 为 7257/3879/2753；F0 的**回合**上限 1500 与**观测窗口** 500 是两个不同数字，play 的 flat 回放按观测窗口 500 步结束，不跑满 1500。注意：训练 env 当前仍调用冻结的 `v02_formal.env.timeout_steps`，与 spec 公式数值一致并由 GPU 逐槽位测试钉住（见 §7.1 D-2）。

## 4. ETH 初始化与迁移

初始化只允许 §1 的 ETH 文件与 SHA，步骤见 SDS §2.1：CPU 校验 → 设置 seed → 构造 `V05ActorCritic` → 完整复制 ETH Actor MLP 与冻结 normalizer → 地图/历史/Critic 按 seed 新建、`fusion.weight/bias=0` → 12 个 std 全部 0.15 → Adam 空、计数 0 → 保存 `migration_zero.pt`（`scope=initialization`，不是 resume checkpoint）。

`V05ActorCritic.from_eth(path, expected_sha, seed, device)` 在迁移脚本、冒烟、worker、evaluate、play 中统一使用；迁移审计证据（A 包）见 §6.2。审计要求：128 组合成观测上与 ETH MLP/归一化绝对误差 ≤1e-6；改地图/历史不改变零融合初始输出；Critic 与同 seed fresh 参考逐 tensor 一致。

## 5. 真实命令

统一 Python：`/home/xxs/anaconda3/envs/a2/bin/python`；headless 设 `MUJOCO_GL=egl`；CPU 命令显式 `CUDA_VISIBLE_DEVICES=`。所有入口在 `import src.tasks` 之前 `activate('scripts/a2_v05_bootstrap.py')`。

### 5.1 CPU 测试（SDS §10 模板）

> **⚠ 必须走本入口，不要直接 `pytest tests/test_v05_*.py`。**
> v0.5 必须在**第一次 import `src.tasks` 之前** activate `scripts/a2_v05_bootstrap.py` 安装的隔离 namespace；直接
> `pytest tests/test_v05_*.py` 会先正常加载 `src.tasks`（触发旧 `import_packages` 全量扫描），bootstrap 随后
> 拒绝替换，**收集期报 4 个 ERROR**（独立验收人实测）。这是隔离机制的硬要求，不是测试缺陷。
> 唯一受支持的命令是下面的 `scripts/test_a2_ideal_stairs_v05.py --pattern 'test_v05_*.py'`，它先 bootstrap 再调 pytest。

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
CUDA_VISIBLE_DEVICES= /home/xxs/anaconda3/envs/a2/bin/python \
  scripts/test_a2_ideal_stairs_v05.py --pattern 'test_v05_*.py'
```

可选：`--verbose`（逐条用例与耗时）、`--log <新文件>`（同一份 transcript 落盘，不覆盖）、`--keyword/-k`、`--maxfail N`。
行为：先 bootstrap，再对 `tests/` 中匹配 `--pattern` 的**已存在**文件运行 pytest；输出逐用例结果、`collected/passed/failed/skipped/errors` 与耗时、退出码（全部通过且有收集到用例才为 0）。不修改、不生成、不跳过测试。单文件示例：

```bash
CUDA_VISIBLE_DEVICES= /home/xxs/anaconda3/envs/a2/bin/python \
  scripts/test_a2_ideal_stairs_v05.py --pattern 'test_v05_reward_analysis.py' --verbose
```

### 5.2 ETH 迁移审计（A 包脚本，真实参数）

该脚本由 A 工作包拥有，本工作包只调用、不重复实现。真实 CLI（读 `scripts/check_a2_ideal_stairs_v05_migration.py` 源码确认）：`--mode{audit}`、`--device{cpu}`、`--seed`、`--project-root`、`--source-checkpoint`、`--source-checkpoint-sha256`、`--output-dir`；脚本内部强制 `CUDA_VISIBLE_DEVICES=''`，`--output-dir` 必须不存在。

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
# seed 42（证据已存在：logs/a2_ideal_stairs_v05/implementation/migration_seed42）
CUDA_VISIBLE_DEVICES= /home/xxs/anaconda3/envs/a2/bin/python \
  scripts/check_a2_ideal_stairs_v05_migration.py --mode audit --device cpu --seed 42 \
  --project-root /home/xxs/dog/unitree_rl_mjlab_a2 \
  --source-checkpoint /home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt \
  --source-checkpoint-sha256 8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722 \
  --output-dir logs/a2_ideal_stairs_v05/implementation/migration_seed42
# seed 43：必须换新目录
CUDA_VISIBLE_DEVICES= /home/xxs/anaconda3/envs/a2/bin/python \
  scripts/check_a2_ideal_stairs_v05_migration.py --mode audit --device cpu --seed 43 \
  --project-root /home/xxs/dog/unitree_rl_mjlab_a2 \
  --source-checkpoint /home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt \
  --source-checkpoint-sha256 8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722 \
  --output-dir logs/a2_ideal_stairs_v05/implementation/migration_seed43
```

### 5.3 run 预检（无 `--execute`：不建 run 目录、不启动 GPU、不联网写入）

pilot42：

```bash
/home/xxs/anaconda3/envs/a2/bin/python scripts/run_a2_ideal_stairs_v05.py \
  --mode pilot --task Unitree-A2-IdealHeightMap-Stairs-V05 --recipe-version 0.5-r1 \
  --sds-revision 1.1 --seed 42 --device cuda:0 --num-envs 512 --eval-num-envs 8 \
  --vx-min 0.2 --vx-max 0.6 --vy 0.0 --wz 0.0 --save-interval 100 \
  --diagnostic-interval 50 --stage-budgets F0=2000,P1=500 --pilot-p1-updates 500 \
  --init-checkpoint /home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt \
  --init-checkpoint-sha256 8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722 \
  --output-dir logs/a2_ideal_stairs_v05/pilot_seed42_trial01
```

pilot43：同上，`--seed 43`、`--output-dir logs/a2_ideal_stairs_v05/pilot_seed43_trial01`。

formal 预检：`--mode formal`、`--seed 42`、去掉 `--pilot-p1-updates`、`--stage-budgets` 写全表
`F0=2000,P1=1000,P2=1000,P3=1000,P4=2000,S1=3000,S2=4000,S3=4000,S4=5000,S5=7000`，
并加两个**真实** pilot 报告：
`--pilot-report-42 logs/a2_ideal_stairs_v05/pilot_seed42_trial01/pilots/pilot_seed42_report.json`、
`--pilot-report-43 logs/a2_ideal_stairs_v05/pilot_seed43_trial01/pilots/pilot_seed43_report.json`，
`--output-dir logs/a2_ideal_stairs_v05/formal_seed42_trial01`。报告路径必须等两个 pilot 真正通过后才存在。

预检产物：`logs/a2_ideal_stairs_v05/implementation/preflight/preflight_<mode>_seed<seed>.json`（含绑定 `behavior_sha256`、`binding_sha256`、`training_started=false`、`gpu_started=false`、`network_started=false`）。目标 run 目录**不会**被创建。

**恢复 run 的预检（`--resume-run`，与 init 互斥）**：输出目录必须已存在且含 `run_contract.json`，其余固定参数与原 run 完全一致（seed/mode/recipe/device/num-envs/budgets 不允许切换）：

```bash
/home/xxs/anaconda3/envs/a2/bin/python scripts/run_a2_ideal_stairs_v05.py \
  --mode pilot --task Unitree-A2-IdealHeightMap-Stairs-V05 --recipe-version 0.5-r1 \
  --sds-revision 1.1 --seed 42 --device cuda:0 --num-envs 512 --eval-num-envs 8 \
  --vx-min 0.2 --vx-max 0.6 --vy 0.0 --wz 0.0 --save-interval 100 \
  --diagnostic-interval 50 --stage-budgets F0=2000,P1=500 --pilot-p1-updates 500 \
  --resume-run --output-dir logs/a2_ideal_stairs_v05/pilot_seed42_trial01
```

`--resume-run` 时**不写** `--init-checkpoint`/`--init-checkpoint-sha256`；加了会被拒绝（互斥）。恢复边界由 `select_resume_boundary` 从 durable ledger 的完整边界选择（不看 mtime），worker 端再由 `--ledger-dir/--ledger-run-id --resume/--resume-sha256` 接续（§5.9）。

独立验收人实测的反例（均为 exit 2，命令可在 `acceptance/ACCEPTANCE_v05.md` §3 第 11–13 行复现）：

- `--resume-run` 与 `--init-checkpoint` 同时给出 → `error: --resume-run and --init-checkpoint are mutually exclusive`。
- `--resume-run` 指向没有 `run_contract.json` 的目录 → `error: Resume run has no run_contract.json`。
- `--resume-run --stage-budgets F0=2000,P1=600` → `error: pilot stage budgets must be exactly F0=2000,P1=500`。
- 无 `--execute` 的预检（含 resume 形态）只写 `implementation/preflight/`，目标目录不会被创建。

### 5.4 GPU 工程冒烟（W6；`--device cuda:0` 才是 GPU 验收）

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
MUJOCO_GL=egl /home/xxs/anaconda3/envs/a2/bin/python scripts/smoke_a2_ideal_stairs_v05.py \
  --task Unitree-A2-IdealHeightMap-Stairs-V05 --stage P1 --num-envs 8 --updates 2 \
  --rollout 48 --epochs 5 --minibatches 4 --seed 42 --device cuda:0 \
  --init-checkpoint /home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt \
  --init-checkpoint-sha256 8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722 \
  --extra-steps 12 \
  --output-dir logs/a2_ideal_stairs_v05/implementation/smoke_gpu
```

CPU dry run（仅工程验证，**不是** GPU 验收；全部参数显式给出）：

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl /home/xxs/anaconda3/envs/a2/bin/python \
  scripts/smoke_a2_ideal_stairs_v05.py \
  --task Unitree-A2-IdealHeightMap-Stairs-V05 --stage P1 --num-envs 8 --updates 2 \
  --rollout 48 --epochs 5 --minibatches 4 --seed 42 --device cpu \
  --init-checkpoint /home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt \
  --init-checkpoint-sha256 8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722 \
  --extra-steps 12 \
  --output-dir logs/a2_ideal_stairs_v05/implementation/smoke_cpu
```

自检项（任一失败即 `status=failed`、退出码 1）：2 更新 / 40 Adam step / 768 控制样本 / 3072 物理子步；奖励、观测、loss、梯度 finite；每步逐项加和误差 ≤1e-5；每次 Adam 后 std∈[0.10,0.30]；`v05-smoke-state-v1` 容器 `formal_resumable=false` 且被 schema6 `preload_checkpoint` 拒绝；额外无学习 F0/P1 混合环境固定动作检查（成本单列，不计入 768）；峰值显存/内存、耗时、退出码落盘。`--device cpu` 结果状态固定为 `passed_cpu_engineering_only`。

**W6 结果**（`implementation/smoke_w6_gpu/smoke.json`，cuda:0）：`status=passed`、40/40 Adam、768/768 控制样本、3072 物理子步、逐项加和误差 0（tol 1e-5）、每次 Adam 后 std∈[0.149869,0.150424]、schema6 拒绝 smoke 容器、无学习 P1(`tiles=[0,1]`)/F0(`tiles=[0]`) 各 96 样本单列、`peak_rss≈3.48 GiB`、`cuda_max_reserved≈60 MiB`。独立验收人另在 cuda:0 实跑两次（17:26 与 17:35 冻结快照，`acceptance/smoke_gpu_audit{,_final}/`）同样 `status=passed`。GPU 环境 + 真实 F0 diagnostic e2e 由验收人在 cuda:0 实跑：**28 passed / 23 subtests / 265.77 s**。

### 5.5 evaluate（冻结模型评估）

必填：`--checkpoint`、`--checkpoint-sha256`、`--stage`、`--kind`、`--seed`、`--num-envs 8`、`--output-dir`（必须不存在）。`--stage` 必须等于 checkpoint 记录的课程阶段；严格更早的阶段必须显式 `--allow-earlier-stage`，结果标记 `acceptance_scope=non_final_earlier_stage`、`final_acceptance=false`；更晚的阶段直接拒绝。`--num-envs` 只允许 8。

```bash
# 模板：<64 位小写 sha256> 必须换成该 checkpoint 的实测 SHA（入口会先 CPU 校验 SHA 再仿真）
/home/xxs/anaconda3/envs/a2/bin/python scripts/evaluate_a2_ideal_stairs_v05.py \
  --checkpoint logs/a2_ideal_stairs_v05/pilot_seed42_trial01/pilot_seed42/checkpoints/update_000500.pt \
  --checkpoint-sha256 <64 位小写 sha256> \
  --stage P1 --kind validation --seed 42 --num-envs 8 --mode pilot --device cuda:0 \
  --output-dir logs/a2_ideal_stairs_v05/manual_eval/pilot42_P1_validation_000500
```

输出：由 `v05.evaluation.evaluate_checkpoint` 写的 `manifest.json`、`episodes.jsonl`、`steps_env0.jsonl`、`window_batch_*.npz`、`report.json`（含 `evidence_verified=true`），以及本入口的 `evaluation_entry.json`（key、acceptance_scope、gate 投影、SHA）。退出码 0 = 评估完成且证据从磁盘重算通过；门禁未通过是数据（`gate_pass=false`），不影响退出码。

`kind`：`validation`（20/速）、`perturbed`（仅 S1）、`diagnostic`（15 例）、`test`（仅 S5、100/速）、`flat_retention`（F0 测 20/速）。`--video` 额外录制每速一份回放视频。

### 5.6 play（单速回放；S5 三速分别成目录）

必填：`--checkpoint`、`--checkpoint-sha256`、`--stage`、`--kind`、`--seed`、`--num-envs 1`、`--vx`（0.2/0.4/0.6，可重复）、`--vy 0.0`、`--wz 0.0`、初态 `--x/--y/--yaw`、`--output-dir`（必须不存在）；`--max-steps` 现在是**可选**的显式硬上限。

`--max-steps` 语义（与 `--help` 和实现一致）：

- 缺省时每速自动取：楼梯阶段 = `spec.episode_step_limit(N, h, vx)`（协议回合上限）；flat（`N=0`）= `spec.F0_OBSERVE_STEPS`（500 步观测窗口）。
- 显式给出时必须 `≥` 上述最小值，否则报错；楼梯不会裁剪 timeout，flat 不会裁剪 500 步观测窗口。
- 每速 `summary.json` 分别记录 `episode_step_cap`（`spec.episode_step_limit` 的值，flat 为 1500）、`observation_window_steps`（flat 为 500，楼梯为 null）、`max_steps` 与 `max_steps_source=cli|default`；`playback_index.json` 记录 `max_steps_by_speed`、`episode_step_caps`、`f0_observation_window_steps`。

每个请求的速度写自己的 `<output-dir>/vx_<speed>/`：`summary.json`、`body_trace.npz`、`vx_<speed>.mp4`（默认录制）。

S5 三速（三条独立命令，三个独立输出目录；`--max-steps` 显式取该速协议回合上限）：

```bash
CK=<S5 checkpoint 路径>; SHA=<对应 64 位 sha256>
for VX in 0.2 0.4 0.6; do
  case $VX in 0.2) MAX=7257;; 0.4) MAX=3879;; 0.6) MAX=2753;; esac
  MUJOCO_GL=egl /home/xxs/anaconda3/envs/a2/bin/python scripts/play_a2_ideal_stairs_v05.py \
    --checkpoint "$CK" --checkpoint-sha256 "$SHA" --stage S5 --kind test --seed 42 \
    --num-envs 1 --mode formal --device cuda:0 \
    --vx $VX --vy 0.0 --wz 0.0 --x -1.0 --y 0.0 --yaw 0.0 \
    --max-steps $MAX --output-dir logs/a2_ideal_stairs_v05/playback/final_S5_vx${VX}
done
```

F0 平地回放（`--max-steps 500` = 观测窗口；`episode_step_cap` 仍是 1500，但回放按 SDS §6.2 的 500 步窗口结束）：

```bash
MUJOCO_GL=egl /home/xxs/anaconda3/envs/a2/bin/python scripts/play_a2_ideal_stairs_v05.py \
  --checkpoint <validated-F0-checkpoint.pt> --checkpoint-sha256 <其 64 位 sha256> \
  --stage F0 --kind flat_retention --seed 42 --num-envs 1 --mode pilot --device cuda:0 \
  --vx 0.2 --vy 0.0 --wz 0.0 --x -1.0 --y 0.0 --yaw 0.0 \
  --max-steps 500 --output-dir logs/a2_ideal_stairs_v05/playback/f0_vx0.2
```

省略 `--max-steps` 时每速自动取默认（楼梯=协议回合上限，flat=500），`max_steps_source=default`；显式给出时
`max_steps_source=cli`，且必须 ≥ 最小值。读结果时请区分：`episode_step_cap`（`spec.episode_step_limit`，
F0=1500、S1@0.4=1266、S5@0.2=7257）是**协议回合上限**；`observation_window_steps`（仅 flat=500）是**观测窗口**；
`max_steps` 是本次回放的实际硬上限；`simulated_steps` 是实际步数。

规则与限制：
- 回放永远不是门禁：`playback_index.json` 与每速 `summary.json` 都写 `gate_eligible=false`、`not_a_gate=true`、`final_acceptance=false`，只报告 `case_pass` 的定义式（F0 500 步+无 failure+横向/航向+speed_pass；P/S 安全到达且非 failure）。
- 指标来自 `v05.speed.cpu_speed_oracle`（F0/P/S 对应 `f0_time`/`p_position`/`s_oracle`），不自行重算；回合上限来自 `spec.episode_step_limit`（单一来源），`--max-steps` 小于最小值直接报错，避免裁剪 timeout / 观测窗口。
- 早期阶段：checkpoint 课程阶段之后的 stage 拒绝；更早的 stage 需 `--allow-earlier-stage`，结果标 `non_final_earlier_stage`。
- **实现限制**：`v05.evaluation` 没有 `replay` kind，且每种 kind 的每速回合数固定（20/100/15），一例回放表会被 `_check_table` 拒绝，所以本入口按 SDS §10 用 `v05.env.V05Env(evaluation=True, fixed_vx, initial_cases)` + `v05.speed.cpu_speed_oracle` 完成回放，而不是调用 `evaluate_checkpoint`。该限制已写入交付报告，未通过伪造 case 表绕过。

### 5.7 诊断分析（§9.3）

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
CUDA_VISIBLE_DEVICES= /home/xxs/anaconda3/envs/a2/bin/python \
  scripts/diagnostics/analyze_a2_v05_rewards.py \
  --output-dir logs/a2_ideal_stairs_v05/implementation/analysis \
  --playback-dir logs/a2_ideal_stairs_v04/playback/vx0.4_sha943ff7e1_20261115_145127 \
  --episodes logs/a2_ideal_stairs_v04/formal_seed42_20261114_234305/formal_seed42/chunk_0000_n512/episodes.jsonl
```

输出 `reward_analysis.json` + `reward_analysis.md`（已存在时需 `--force` 才替换这两个文件；`--no-traces` 只跳过 trace 重算）。内容：7 个固定反例（静止/低速拖行/超速撞击/前倾不落脚/重复抬脚/正常有效上阶/提前摔倒，20 s/1000 步/c=0.4/γ=0.995）的逐项、总和与折扣和；SDS 锚点 3.973384 / -2.482749 / -6.839384（实测在 1e-6 内）；3×6×5=90 点时间伸缩网格（swing 全领/关闭/只落奇数阶 × 连续正则率 0/-0.25/-0.5）；0.4 停车 trace 与初始高速失败 trace 的可重构项、以及两份**显式缺失字段**清单。离线折扣**不含** timeout bootstrap，报告已注明。

### 5.8 W7～W10（已获授权；工程验收已通过；W7 首次执行未通过、W8–W10 未执行）

按 SDS 顺序，工程验收（W0～W6）通过后再执行；命令与参数必须显式，不得省写默认值：

- W7 资源准入：`run --mode probe`（supervisor 的 W7 执行链，`orchestration.run_resource_probe`）——5 更新 warmup + 50 更新测量 + 1 更新恢复，另做一次 8 环境 P1 评估生命周期（训练 env/model 常驻）与 S5 三速全超时无学习 probe，最后从 update-55 边界恢复；成本独立记账、不进 formal/pilot 预算。完整命令（全部参数显式）：

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
# 预检形态（无 --execute）：只写 implementation/preflight/preflight_probe_seed42.json，不建 run 目录
/home/xxs/anaconda3/envs/a2/bin/python scripts/run_a2_ideal_stairs_v05.py \
  --mode probe --task Unitree-A2-IdealHeightMap-Stairs-V05 --recipe-version 0.5-r1 \
  --sds-revision 1.1 --seed 42 --device cuda:0 --num-envs 512 --eval-num-envs 8 \
  --vx-min 0.2 --vx-max 0.6 --vy 0.0 --wz 0.0 --save-interval 100 \
  --diagnostic-interval 50 \
  --stage-budgets F0=2000,P1=1000,P2=1000,P3=1000,P4=2000,S1=3000,S2=4000,S3=4000,S4=5000,S5=7000 \
  --init-checkpoint /home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt \
  --init-checkpoint-sha256 8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722 \
  --output-dir logs/a2_ideal_stairs_v05/w7_probe_trial01

# 执行形态：在完全相同的参数上追加 --execute（--output-dir 必须不存在）
```

    probe 契约（由 `orchestration.validate_arguments` 断言，`run` 脚本原样转发、不重复校验）：`--seed 42`、**完整 formal 十阶段预算表**、禁止 `--pilot-p1-updates`/`--pilot-report-*`、禁止 `--resume-run`（一次性资源运行不可恢复）；`--output-dir` 必须不存在（既存目录会立刻 exit 2）。产物：`<output-dir>/probe/probe.json`（含 `probe_stage=P1` 与理由、阈值、成本、失败项）、`probe/resources.jsonl`（每控制步采样）、`probe/summary.md`、`probe/ledger/`、`probe/checkpoints/probe_boundary_000055.pt`；probe 不建 W&B 会话（`wandb_session=none_for_probe`），训练 cadence 关闭（诊断/门禁/保存另计）。
    **probe 阶段固定 P1 的理由**（`orchestration.PROBE_STAGE_REASON`）：SDS §7.5/§8.2 未固定 probe 阶段；P1 是第一个具备 512 环境固定槽位配比（102 flat + 410 current）的楼梯阶段，其驻留训练内存/显存占用对后续楼梯阶段有代表性；长回放时间包络由单独的无学习 S5 三速全超时 probe 覆盖，不需要在 probe 里跑 S5 学习。
    **W7 首次真实执行（2026-11-15 18:39，`logs/a2_ideal_stairs_v05/W7_probe_seed42_20261115_183947/`）：未通过。** 纯 PPO 段成功（5 warmup + 50 measured = 55 更新、1,351,680 控制样本、5,406,720 物理子步、1,100 Adam），四项资源门槛全过（整卡峰值 5121.25/7168 MiB、进程 3188/5120 MiB、host 10245.1/4096 MiB、磁盘 469.0/10 GiB、中位更新 5.65 s）；但**共存评估阶段**报 `ValueError: evaluation mode must be pilot or formal`，recovery 与 S5 三速段未执行（`probe.json: status=failed`）。
    **修复状态**：`v05/evaluation.py` 现已支持 `mode='probe'`（`EVALUATION_MODES` 与 `check_evaluation_mode`，probe 仅允许 `validation/flat_retention/diagnostic` 等非门禁 kind），`v05/orchestration.py` 也改为以 `mode='probe'` 提交共存评估；**W7 尚未重跑，本文不把 W7 记为通过**。共存评估依赖修复后的 `evaluate_checkpoint(mode='probe')`，因此重跑前须确认这两处已在冻结源码中。
- W8 两个 pilot：两条 `run ... --mode pilot --execute` 命令，seed 42 与 43 串行、各自从 ETH 重新初始化、各自新目录；`F0=2000,P1=500`，P1 固定跑满 500 更新。
- W9 formal：`run --mode formal --execute`，seed 42、cuda:0、512/8 环境、全阶段表预算、绑定两个**真实通过**的 pilot 报告；从 ETH 重新开始，不继承 pilot 权重。
- W10：冻结第二次 S5 通过的模型 → final test（400342，三速各 100，`case_pass ≥95/100`）→ 三速回放（§5.6）→ CPU 复算（`v05.speed.recompute_npz`）→ 报告。

授权状态：**已获授权；工程验收（W0～W6/GPU smoke/独立验收/Q002 签署）已通过**；**W7 已首次执行但未通过（共存评估 mode 问题，已修复待重跑），W8–W10 未执行**。Q002 签署已解除 Q001/Q002 物理等价项对 W8/W9 的阻塞，但 **W7 完成仍是 W8 前的 SDS 前置条件**。启动前必须确认源码已冻结并按 SDS 重新生成 W6+ 证据。任一门槛失败按 SDS 停止/暂停并报告，不自行加预算、换配方或挑选旧模型。

### 5.9 worker 合同（supervisor 生成；不供人手调用）

`Supervisor.worker_command()` 生成的命令形状（pilot / formal / resume 三种）：

```text
python scripts/train_a2_ideal_stairs_v05.py   --task Unitree-A2-IdealHeightMap-Stairs-V05 --mode pilot|formal --seed 42|43   --device cuda:0 --num-envs 512 --eval-num-envs 8   --vx-min 0.2 --vx-max 0.6 --vy 0.0 --wz 0.0   --save-interval 100 --diagnostic-interval 50   --stage-budgets F0=2000,P1=500 [--pilot-p1-updates 500]   [--init-checkpoint <ETH> --init-checkpoint-sha256 <ETH_SHA> |    --resume <checkpoint> --resume-sha256 <64hex>]   --output-dir <trial> --ledger-dir <trial>/ledger   --ledger-run-id v05-{mode}-seed{seed} --control-file <supervisor control.json>
```

规则（实现见 `scripts/train_a2_ideal_stairs_v05.py`，`--help` 已列出全部参数）：

- `--init-checkpoint/--init-checkpoint-sha256` 与 `--resume/--resume-sha256` **互斥**，且必须恰好给出一组；`--init-checkpoint-sha256` 只能是冻结 ETH SHA。
- `--ledger-dir` 与 `--ledger-run-id` 必填；ledger 文件固定为 `<ledger-dir>/ledger.jsonl`，本次 trial 的每个 checkpoint 都由**同一** ledger 的 `record_checkpoint` 提交（`result.json` 的 `ledger` 块给出 path/anchor）。
- `--ledger-run-id` 必须等于 `v05-{mode}-seed{seed}`，否则参数校验即报错（防止串 run）。
- resume 时输出目录必须已存在；若 `<ledger-dir>/ledger.jsonl` 不存在或 `V05Runner.resume` 抛出 `RecoveryBlocked`，worker 写 `failure.json`（`status=recovery_blocked`）并以非 0 退出，**不会**回退到 ETH 初始化继续训练。
- resume 的模型先按 ETH 迁移只构造网络形状，随后由 checkpoint 的完整 `model`/`optimizer`/学习计数覆盖；失败即中止，不存在“init 顶上”的路径。

恢复 worker 命令（supervisor 生成；以下为 pilot42 的具体形态，`<checkpoint>`/`<sha256>` 由 `select_resume_boundary` 选定）：

```bash
/home/xxs/anaconda3/envs/a2/bin/python scripts/train_a2_ideal_stairs_v05.py \
  --task Unitree-A2-IdealHeightMap-Stairs-V05 --mode pilot --seed 42 --device cuda:0 \
  --num-envs 512 --eval-num-envs 8 --vx-min 0.2 --vx-max 0.6 --vy 0.0 --wz 0.0 \
  --save-interval 100 --diagnostic-interval 50 \
  --stage-budgets F0=2000,P1=500 --pilot-p1-updates 500 \
  --output-dir logs/a2_ideal_stairs_v05/pilot_seed42_trial01/pilot_seed42 \
  --ledger-dir logs/a2_ideal_stairs_v05/pilot_seed42_trial01/pilot_seed42/ledger \
  --ledger-run-id v05-pilot-seed42 \
  --resume <checkpoint> --resume-sha256 <64 位 sha256> \
  --control-file logs/a2_ideal_stairs_v05/pilot_seed42_trial01/control.json
```

（新 trial 形态把 `--resume/--resume-sha256` 换成
`--init-checkpoint /home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt --init-checkpoint-sha256 8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722`；两对参数互斥。）

## 6. 本工作包已执行的验证（含真实输出）

### 6.1 诊断分析

`scripts/diagnostics/analyze_a2_v05_rewards.py`（CPU）输出：
`matched_speed=3.973384`、`stationary=-2.482749`、`early_failure=-6.839384`（与 SDS 锚点差 <1e-6）；
7 个反例三条必要条件全部通过（匹配速度 > 静止、重复抬脚额度 = 0.600000 ≤0.60、提前失败折扣和 -6.839384 < 静止 -2.482749）；
时间伸缩 90 点；N=40/c=0.4 下高于匹配速度的 v/c 实测为 `1.2/1.5/2.0`（SDS 已承认 1.5/2.0，即 0.6/0.8 m/s；本分析另外测到 1.2，即 0.48 m/s）；`blocking_for_0_5=false`，报告明确“不声称本版速度最优”。
0.4 停车 trace：1266 控制步、`c=0.4`、timeout、`stall` 首次扣费 t=99、可重构项合计（折扣）约 `-1.481287`；swing 状态**未重算**，缺失字段 16 项。
初始高速失败 trace：0～9 更新内 5124 例失败回合，平均 `c≈0.3981`、平均 `vx≈1.4356`、平均超速比≈4.04、失败原因以 `nonfoot_persistent` 为主；逐 step 序列与 swing/stall 状态缺失，缺失字段 8 项。

### 6.2 CPU 测试

```bash
CUDA_VISIBLE_DEVICES= /home/xxs/anaconda3/envs/a2/bin/python \
  scripts/test_a2_ideal_stairs_v05.py --pattern 'test_v05_*.py' \
  --log logs/a2_ideal_stairs_v05/implementation/cpu_tests/final_frozen.log
```

`tests/test_v05_reward_analysis.py` 单独运行结果：`12 passed`（覆盖 7 反例三条必要条件、3×6×5 网格与折扣复算、缺失字段显式化、同输入同输出可重复性、SDS 三个锚点）。

2026-11-15 18:39（官方入口、冻结源码）：**394 收集 / 373 通过 / 21 跳过 / 0 失败 / 179 subtests / 184.1 s，退出码 0**，日志 `implementation/cpu_tests/final_frozen.log`；21 skipped = 20 个 GPU 环境用例 + 1 个 GPU 真实环境评估 e2e 用例。历史上 17:35 的 370/353/17/185.1 s 与 16:43 的 345/323/6 均为旧快照，保留以免把“当时失败”写成“从未失败”。

同日 16:43 曾有一次运行得到 345 收集 / 323 通过 / 6 失败（失败全在 C 包 `test_v05_orchestration.py`、`test_v05_runner.py`）；C 包在 16:58 前完成修复后复跑全部通过，本文保留该过程记录以免把“当时失败”写成“从未失败”。

GPU-only 用例（cuda:0，**由独立验收人执行**）：`tests/test_v05_environment.py` + `tests/test_v05_evaluation.py::RealEnvironmentEndToEndTests` 合计 **28 passed / 23 subtests / 265.77 s**（`Q002_acceptance_addendum.md` §5 第 2 行；含真实 `V05Env`×`V05ActorCritic` 的 15 例 F0 诊断端到端）。CPU 全套里这些用例被 skip（21 skipped），因此它们不是“只 skip 不验证”。

### 6.3 预检

`--mode pilot --seed 42`（§5.3 命令，无 `--execute`）退出码 0，写
`logs/a2_ideal_stairs_v05/implementation/preflight/preflight_pilot_seed42.json`；
目标目录 `logs/a2_ideal_stairs_v05/pilot_seed42_trial01` **未被创建**；记录内 `training_started=false`、`gpu_started=false`、`network_started=false`。

### 6.4 CPU 冒烟（工程 dry run）

`--device cpu` 冒烟通过：40/40 Adam step、768/768 控制样本、3072 物理子步、逐项加和误差 0、40 次 Adam 后 std ∈[0.149743,0.150515]、schema6 preload 拒绝 `v05-smoke-state-v1`、无学习 P1（tile `[0,1]`）与 F0（tile `[0]`）混合检查各 96 样本单独计费。状态 `passed_cpu_engineering_only`，**不是** GPU 验收。

### 6.5 结果登记（已执行项与证据索引）

| 项 | 命令/入口 | 结果 | 证据 |
|---|---|---|---|
| CPU 测试全套（官方入口） | §6.2 | **394 收集 / 373 通过 / 0 失败 / 21 跳过 / 179 subtests，184.1 s，退出码 0** | `implementation/cpu_tests/final_frozen.log` |
| CPU 冒烟 | §5.4（`--device cpu`） | `passed_cpu_engineering_only`（40 Adam、768 样本、std 0.149743–0.150515、preload 拒绝 smoke 容器） | `logs/a2_ideal_stairs_v05/implementation/smoke_cpu/smoke.json` |
| W6 GPU smoke | §5.4（`--device cuda:0`） | `status=passed`、40/40 Adam、768/768 控制样本、3072 物理子步、加和误差 0、std∈[0.149869,0.150424]、schema6 拒绝 smoke 容器；验收人两次独立复跑同样 passed | `implementation/smoke_w6_gpu/smoke.json`、`acceptance/smoke_gpu_audit{,_final}/` |
| GPU 环境 + F0 e2e | `test_v05_environment.py` + `test_v05_evaluation.py::RealEnvironmentEndToEndTests`（cuda:0） | **28 passed / 23 subtests / 265.77 s**（独立验收人） | `acceptance/Q002_acceptance_addendum.md` §5 |
| ETH 迁移审计 42/43 | `check_a2_ideal_stairs_v05_migration.py --mode audit --device cpu` | 两个 final 目录均 `status=passed`、`max_abs_error=0.0`、`cuda_initialized=false`、51 tensor 行 | `implementation/migration_seed42_final/`（zero `d08b6d84…`）、`migration_seed43_final/`（zero `8f88c2e8…`） |
| W7 资源 probe（首次） | `run --mode probe --execute` | **未通过**：55 更新成功、四项资源门槛全过；共存评估 `evaluation mode must be pilot or formal`；recovery/S5 未执行；已修复待重跑 | `logs/a2_ideal_stairs_v05/W7_probe_seed42_20261115_183947/probe/probe.json` |
| pilot42/43 预检 | §5.3 | 两次均 `preflight_validated`、exit 0、未建 run 目录（`source_sha=b30cf492…`、`binding_sha256=6ba289f2…`） | `implementation/preflight/preflight_pilot_seed4{2,3}.json` |
| `--resume-run` 反例 | §5.3 | 三条均 exit 2（互斥 / 缺 run_contract / 预算不符） | `implementation/acceptance/ACCEPTANCE_v05.md` §3 第 11–13 行 |
| evaluate 实跑 | §5.5 | F0/validation（20/速）与 F0/diagnostic（15 例）均已跑通，`evidence_verified=True`；门禁计数 0/20、10/20、11/20（0 更新模型，属预期失败） | `implementation/wiring_checks/` |
| play 实跑 | §5.6 | F0/vx0.2/500 步已跑通（接线验证）；S5 三速待有 S5 checkpoint 后执行 | `implementation/wiring_checks/play_f0_vx02/` |

### 6.6 evaluate / play 接线验证（CPU，临时 0 更新 schema6 checkpoint）

用 `V05ActorCritic.from_eth` + `V05Runner.save_checkpoint('stage_start')` 在临时目录生成一个 0 更新 schema6 checkpoint（不训练、不占用 run 目录；验证后已删除 checkpoint 与 npz，仅保留 `implementation/wiring_checks/` 下的入口记录 JSON），再按 §5.5/§5.6 的命令执行：

- `evaluate --stage F0 --kind validation --num-envs 8 --device cpu`：9 个 batch（20/速）全部完成，`report.json` 的 `evidence_verified=True`、`gate_eligible=True`、`required=18`，三速 `case_pass=0/20、10/20、11/20`（0 更新策略的预期失败），`evaluation_entry.json` 写入 key/SHA/acceptance_scope。耗时约 8 分钟。
- `evaluate --stage F0 --kind diagnostic --num-envs 8 --device cpu`：15 例、`per_case` 可读、`diagnostic_decision` 可判（见 §7.2 已解决项）。
- `play --stage F0 --kind flat_retention --vx 0.2 --max-steps 500 --no-video --device cpu`：500 步、`window_kind=f0_time`、`samples=400`、`speed_valid=True`、`speed_pass=False`、`gate_eligible=false`，输出 `vx_0.2/{summary.json,body_trace.npz}` + `playback_index.json`，模型张量前后一致。
- `evaluate` 的负向路径：缺 `--num-envs`/`--output-dir` 报 argparse 错误；`--output-dir` 已存在报错；错误 SHA 报 `Checkpoint SHA mismatch`；`--stage` 晚于 checkpoint 课程阶段被拒。

`play --max-steps` 缺省路径也已验证：F0 自动取 500 步观测窗口（`episode_step_cap=1500`、`observation_window_steps=500`、`max_steps_source=default`），楼梯阶段自动取 `spec.episode_step_limit`（S1=2031/1266/1011、S5=7257/3879/2753），显式给小于最小值的 `--max-steps` 会被拒。

worker 命令合同（§5.9）验证：用 `Supervisor.worker_command()` 真实生成的 **pilot42 / pilot43 / formal42 / resume** 四条命令逐条喂给 `train` 的 parser+validate，全部接受（resume 命令含 `--resume/--resume-sha256`、不含 init；其余含 init、不含 resume；`--ledger-dir`/`--ledger-run-id` 均正确）。四条负向均被拒：resume+init 同给、resume 缺 sha、缺 `--ledger-dir`、`--ledger-run-id` 与 mode/seed 不符。另用真实 worker 进程验证 fail-closed：`--resume` 指向不存在的 `<ledger-dir>/ledger.jsonl` 时退出码 1、`failure.json` 的 `status=recovery_blocked`，`init_checkpoint=null`，没有回退到 ETH 继续。

这些结果只证明“入口与真实 v05 API 接线正确”，不构成任何训练效果或门禁结论；临时 checkpoint 不是交付证据（源码闭包随并行编辑变化，见 §0 的源码闭包说明）。

### 6.7 独立验收人执行的关键运行（引自 `acceptance/ACCEPTANCE_v05.md` 与 `Q002_acceptance_addendum.md`）

| # | 运行 | 结果 |
|---|---|---|
| 1 | CPU 全套（官方入口，冻结源码，18:39） | 394 collected / **373 passed / 21 skipped / 0 failed** / 179 subtests / 184.01 s，exit 0 |
| 2 | GPU 环境 + 真实 F0 diagnostic e2e（cuda:0，18:35–18:55） | **28 passed / 23 subtests / 265.77 s**（27 环境用例 + 15 例真实 `V05Env`×`V05ActorCritic` F0 诊断端到端） |
| 3 | GPU smoke（W6，cuda:0） | `status=passed`：40/40 Adam、768/768 控制样本、3072 物理子步、加和误差 0、std `[0.149869,0.150424]`、schema6 拒绝 smoke 容器、无学习 P1/F0 各 96 样本 |
| 4 | Q001/Q002 物理等价证据链 | canonical `summary.json`：`status=failed_q001`、`long_horizon_numeric_equivalence=not_established`；`summary_q002.json::admission=accepted_with_known_limitation_Q002`；canonical 4 文件 SHA 未被改写；独立复算前缀 25/20/7、前缀内违规 0 |
| 5 | 迁移审计 42/43（CPU） | 两 seed `status=passed`、`max_abs_error=0.0`、`cuda_initialized=false` |
| 6 | 预检 pilot42/43（无 `--execute`） | `preflight_validated`、exit 0、目标目录未创建（`source_sha=b30cf492…`、`binding_sha256=6ba289f2…`） |
| 7 | `--resume-run` 三条反例 | 均 exit 2：`--resume-run`+`--init-checkpoint` 互斥 / 缺 `run_contract.json` / `--stage-budgets F0=2000,P1=600` 被拒 |
| 8 | 闭包/隔离与 fail-closed 探针 | 74 文件闭包、`src/tasks/__init__.py` 排除、`sys.modules['src.tasks'].__file__ is None`；13 个 schema/配方/协议/单位反例全部抛错 |

## 7. 已知限制与偏差（以独立验收与 Q002 补充报告为准；阻断级 0 项）

依据：`implementation/acceptance/ACCEPTANCE_v05.md`（17:30 快照）与 `Q002_acceptance_addendum.md`（18:35–18:55，
Q001/Q002 裁决后签署）。验收首轮列出非阻断偏差 7 项（P2×2、P3×5）+ 1 项交付缺口；其中 D-1、D-3、D-4、D-7 已修复或勘误，
seed43 迁移审计已补齐（`migration_seed43_final`），Q001/Q002 物理等价项已由补充报告签署（**不再阻塞 W8/W9**）。
本节只列**当前仍存在**的项与运行侧注意事项。

### 7.1 当前仍存在（非阻断）

1. **D-2（B 包，P2）训练 timeout 未走 `spec` 单一来源**：`v05/env.py:43,360` 仍 import/调用冻结的
   `v02_formal.env.timeout_steps`，而 `spec.stair_episode_steps/episode_step_limit` 只被 `v05/evaluation.py` 与
   `scripts/play_a2_ideal_stairs_v05.py` 使用。**数值一致**由 GPU 测试逐槽位钉住（F0/P1/S1/S4/S5 × c=0.2 的真实 env
   timeout = `spec.episode_step_limit`；`tests/test_v05_evaluation.py` 逐阶段×逐速度比对），风险仅是未来单侧修改后漂移；
   建议 env 直接改调 spec 或把测试扩到 10 阶段 × 3 速度（Q002 §7-R5）。
2. **canonical 证据缺失时测试为 skip（fail-open）**：`Q002CanonicalEvidenceTests._load()` 在 `summary.json` 不存在时
   `skipTest` 而非失败，canonical 证据被删只会 skip、`summary_q002.json` 仍保留旧 SHA。建议冻结 CI 把
   “canonical 4 文件存在且 SHA 等于登记值”改成硬断言（Q002 §7-R1）；另有 R2（live 终止帧非空只记录不断言）、
   R3（每步各推进一次由静态调用顺序断言替代运行期计数）两项观察，均不影响当前签署。
3. **`evaluation` 没有 `replay` kind 属设计选择（非缺陷）**：`v05/evaluation.py:68` 的 `KINDS` 不含 `replay`，
   `_expected_episodes` 固定每速 20/100/15 且 `_check_table` 要求恰好三档速度，因此 `evaluate_checkpoint` 无法表达
   SDS §10 的“单速一回放”。`scripts/play_a2_ideal_stairs_v05.py` 用真实 `V05Env(evaluation=True, fixed_vx,
   initial_cases)` + `v05.speed.cpu_speed_oracle` 实现回放并标 `gate_eligible=false`/`not_a_gate=true`/
   `final_acceptance=false`；验收与 Q002 补充报告均未把它列为偏差。若将来要求回放统一走 `evaluate_checkpoint`，
   需 B 包新增 `kind='replay'`。
4. **文档卫生（不影响判定）**：`CONTRACT_v05.md:43` 原文仍写“S1 无 previous 槽”，同文件 `:166` 已勘误并以 SDS/实现为准；
   `v05/runner.py:218` 的 `PilotLedgerConflict` 类 docstring 仍写“C1 interface gap”（`_require_running` 的 raise 文案已更新）。
5. **运行侧注意事项（本工作包）**：CPU 上 8 环境×20 例/速完整门禁约 8 min（S1–S5 更长），正式门禁仍在 GPU；
   `run` 预检会把完整 `binding` 打到 stdout（记录同时落盘）；worker 把 durable JSONL/JSON 的 nan/±inf 写成 `null`
   并记 `sanitized_nonfinite_floats`（B 修复后应为 0）。

### 7.2 已解决（历史留档，勿再按旧版本文档排障）

- **P0（本工作包发现）`V05Runner.reset_episode`/`resume` 未解包 `env.reset()` 的 `(obs, info)`**：曾使第一次
  `collect()` 抛 `AttributeError: 'tuple' object has no attribute 'items'`（本工作包在 18:10 的 CPU 复现与一次误触发的
  probe 短跑中暴露）；`runner.py` 现由 `_environment_reset()` 统一返回 `(obs, info)`，两处调用均已修正（C2，18:19），
  18:39 的 W7 首跑已能完成 55 个更新，证明该 P0 已消除。
- **W7 共存评估 mode**：首跑失败于 `evaluation mode must be pilot or formal`；`v05/evaluation.py` 现支持
  `mode='probe'`（`EVALUATION_MODES` + `check_evaluation_mode`，probe 仅允许非门禁 kind），`orchestration.py` 以
  `mode='probe'` 提交共存评估（B/C，18:51）。**W7 尚未重跑**，不得记为通过。
- **D-1（验收 P2）resume 二次 reset**：`runner.resume` 现在**先** `set_stage` 绑定课程/槽位、**再** `restore_rng`，
  并断言 restore 不 reset、RNG 恢复后恰好一次显式 reset（Q002 §7-R6 复核修复）。
- **D-3（验收 P3）状态推进顺序**：`v05/env.py:374-376` 现为 footfall→swing→stall，与 SDS §4 一致。
- **D-4（验收 P3）**：`CONTRACT_v05.md:166` 已勘误（原文待改写）；**D-7** 两个 SHA 命名杂项文件已删除；
  `migration_seed43_final` 已补齐并 `status=passed`。
- **A 包**：`v05/rewards.py` 的 flat(N=0) finite 断言只在 `required` 行断言（P1 混合槽位真动作 120 步回归通过）。
- **B 包**：`v05/evaluation.py` 的 `jsonable()` 递归把 nan/±inf 映射为 `None` 并计数（`evaluation.py:88-114`）。
- **B/C 接缝**：每 50 更新诊断的 `diagnostic_cases`/`per_case` 键名与 F0 合成终态计数已补齐，`diagnostic_decision` 可判；
  真实 `V05Env`×`V05ActorCritic` 的 F0 diagnostic e2e 已进入 GPU 套件（28 passed 内含）。
- **A 包**：`spec` 现导出 `stair_episode_steps`/`episode_step_limit`，`play` 已改为单一来源；`evaluation._protocol_step_limit`
  与 spec 对齐。
- **C 包**：supervisor 支持 `--resume-run`（run contract 逐字段校验 + ledger 完整边界 + 与 init 互斥）；worker 侧
  `--ledger-dir/--ledger-run-id/--resume/--resume-sha256` 已接线并 fail-closed。
- **Q001/Q002 物理等价**：Q001 裁决后 (a′) 实测 failed，`failed_q001` 与 `not_established` 保留；Q002 裁决接受有限证据链，
  `admission=accepted_with_known_limitation_Q002`（详见 §8.1）。

## 8. 独立验收结论（引用 `acceptance/ACCEPTANCE_v05.md` 与 `Q002_acceptance_addendum.md`）

- **报告链**：`ACCEPTANCE_v05.md` 基于 2026-11-15 **17:30 冻结快照**（17:36 落稿）；`Q002_acceptance_addendum.md`
  （18:35–18:55）是 Q002 裁决后的追加签署，**纠正**了首轮把 SDS §9.2-11 记为“passed（带歧义）”的结论，并保留历史报告不改写。
  本文件 §7.2 另记录快照之后由责任方修复、本轮核对为已修复的项（含 P0 reset_episode、W7 共存评估 mode）。
- **结论**：有条件交付 —— **工程验收通过、训练效果未验证**；首轮判定**阻断级偏差 0 项**、非阻断 7 项 + 1 交付缺口；
  其中 D-1/D-3/D-4/D-7 已修复或勘误、seed43 迁移审计已补齐，Q001/Q002 物理等价项已签署且**不再阻塞 W8/W9**；
  当前仍存在项见 §7.1。
- **验收人本机复跑的关键数字**（Q002 签字快照）：CPU 全套 394 collected / **373 passed / 21 skipped / 0 failed**
  （179 subtests / 184.01 s，exit 0）；GPU 环境 + 真实 F0 diagnostic e2e **28 passed / 23 subtests / 265.77 s**（cuda:0）；
  W6 GPU smoke `status=passed`（40/40 Adam、768/768 控制样本、3072 物理子步、加和误差 0、std∈[0.149869,0.150424]、
  schema6 拒绝 smoke 容器）；迁移审计 42/43 均 `passed`（`max_abs_error=0.0`、未初始化 CUDA）。
- **预检 / `--resume-run` 反例**：pilot42/43 预检均 `preflight_validated`、exit 0、未建 run 目录
  （`source_sha=b30cf492…`、`binding_sha256=6ba289f2…`）；三条 `--resume-run` 反例均 exit 2（互斥 / 缺 `run_contract.json` /
  `--stage-budgets F0=2000,P1=600`）。
- **W7 首次真实执行未通过**（55 更新成功、四项资源门槛全过；共存评估报 `evaluation mode must be pilot or formal`；
  recovery/S5 未执行）。修复已落地但**尚未重跑**，因此 W7 不能记为通过，W8 的前置条件仍未满足（见 §5.8）。
- **仍需处理（非阻断）**：§7.1 的 D-2（timeout 走 spec）、canonical 证据 fail-open 的 CI 硬断言、`replay` kind 的设计选择，
  以及验收报告 §5 的 3 条规范歧义（A-1 新旧等价容差、A-2 float32 边界口径、A-3 dx 条件冗余）交决策方裁决。
- **工程通过 ≠ 训练有效**：本文件不包含任何爬楼成功率、速度达标或“已解决停车”的结论；W8 pilot、W9 formal、
  W10 final test 均未执行，这些只能由后续真实运行给出，且必须按 SDS 在源码冻结后重新生成证据。

### 8.1 Q001/Q002 物理等价裁决与证据链（Q002 §4.3）

**文件与关系**（按裁决顺序）：

| 文件 | 作用 |
|---|---|
| `implementation/coordination/Q001.md` | 实施方向决策方提出的问题：SDS §9.2-11 的 `1e-5`/100 步判据在 MuJoCo-Warp 上不可执行；stall 边界的 float32 等值口径 |
| `implementation/coordination/Q001_decision.md` | 决策方裁决：有条件接受修正后的 (a′)（只改测试、判定集中于单一 `q001_evaluate()`；`old_spread`/`bound`/选帧只用旧-旧；前缀内逐帧断言；三种子独立构造）；stall 维持 float32 严格小于 |
| `implementation/acceptance/q001_physics_equivalence/` | (a′) 实测证据：`summary.json`（canonical，**未被改写**）与逐帧 `seed_0/42/43.json`；结果 `status=failed_q001`、`long_horizon_numeric_equivalence=not_established`。另有 Q002 追加证据 `summary_q002.json`、`q002_*.json`，以及补充诊断 `reference_diagnostic.json` |
| `implementation/coordination/Q002.md` | 实施方在 (a′) 实测 failed 后再次提交：同一实现/同实例重放也出现数值分歧，请求以有限证据链准入 |
| `implementation/coordination/Q002_decision.md` | 决策方裁决：接受修订 A′，采用 `long_horizon_numeric_equivalence=not_established`（不采用 `not_testable_engine_nondeterminism`），并明确 W8/W9 的解除条件（须由独立验收签署补充报告） |
| `implementation/acceptance/Q002_acceptance_addendum.md` | 独立验收补充报告（18:35–18:55）：四项证据 (i)–(iv) 全部通过、canonical 未被改写、测试未弱化，**签署放行**；同时纠正首轮 §9.2-11 的“passed”结论 |

**状态三元（三者并存，互不矛盾，均以 canonical 文件为准）**：

| 字段 | 值 | 位置 |
|---|---|---|
| `admission`（准入） | `accepted_with_known_limitation_Q002` | `acceptance/q001_physics_equivalence/summary_q002.json` |
| `long_horizon_numeric_equivalence`（长期数值等价） | `not_established` | 同上与 canonical `summary.json` |
| `q001_history` / `q001_canonical_status`（历史） | `failed_q001` | 同上与 canonical `summary.json::status` |

**证据不变量**：

- canonical 四文件（`summary.json`、`seed_0.json`、`seed_42.json`、`seed_43.json`）**不得改写**，SHA 登记在
  `summary_q002.json::q001_canonical_files` 与补充报告 §6；补充报告签认 4/4 一致（`summary=b80c9907…`、
  `seed_0=56277e34…`、`seed_42=63b0be08…`、`seed_43=6c985b21…`），删除或择优重试都不允许——否则即失去
  `failed_q001` 的历史事实。注意“canonical 证据被删除时测试只会 skip”仍是 fail-open（§7.1 第 2 条）。
- `reference_diagnostic.json` 是**补充诊断**（`supplementary=true`、`replaces_q001=false`、不在 canonical SHA 清单、
  不参与准入判定），**每次 GPU 运行都会重写**（其 SHA 会变，例如 `1c03a767…` → `23c75dd4…`）；它只说明
  “v0.5 跟踪到 step 0–1 逐位相同、step 2 起出现 ~1.5e-5 差异、到 step 10 两个 stock V04Env 之间已达 ~5.3e-5”，
  即分叉并非 v05 独有，不能当作替代 Q001 的等价证据。
- 本文件与 SDS 原文、历史验收报告均**不追改**；Q001/Q002 只追加裁决、追加证据与追加签署。

## 9. 复现与环境

- 仓库：`/home/xxs/dog/unitree_rl_mjlab_a2`（git main，工作区含大量未提交改动；本工作包不 commit、不清理、不覆盖旧文件）。
- Python：`/home/xxs/anaconda3/envs/a2/bin/python`（3.11.18，torch 2.15.0+cu132，pytest 9.2.0）。
- headless：`MUJOCO_GL=egl`；CPU 命令显式 `CUDA_VISIBLE_DEVICES=`。
- **不要**在 `/tmp` 下运行脚本（`/tmp/profile.py` 会遮蔽标准库 `profile`）。
- 证据目录：`logs/a2_ideal_stairs_v05/implementation/{preflight,analysis,cpu_tests,smoke_cpu,wiring_checks,migration_seed42,acceptance}/`；验收报告见 `implementation/acceptance/ACCEPTANCE_v05.md`。
