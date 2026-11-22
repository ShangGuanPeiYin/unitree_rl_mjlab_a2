# A2 理想高程图镂空楼梯 v0.4 实施与操作说明

- 代码版本：`0.4.0`；奖励配方：`0.4-r1`；checkpoint schema：`5`。
- SDS：`doc/sds/SDS_A2_IdealHeightMap_Stairs40_v0.4.md`（修订 1.1）；独立审核记录：`doc/sds/reviews/SDS_v0.4_review_round1.md`。
- 本文范围：V0 来源冻结、V1 奖励/导入隔离、V2 环境接线、V3 模型/schema5/runner、V4 评估/监控、V5a 8 环境×2 更新 GPU 冒烟、V7 交付文档。
- 未执行：512 环境资源生命周期 probe、0 更新基线、正式课程训练、正式 S1–S5 长训和最终成功率验收。本文不把单测、合成轨迹或冒烟通过写成爬楼效果改善。

## 1. 版本身份与冻结参数

| 项 | 值 |
|---|---|
| `CODE_VERSION` | `0.4.0` |
| `RECIPE_VERSION` | `0.4-r1` |
| `SCHEMA_VERSION` | `5` |
| `TASK` | `Unitree-A2-IdealHeightMap-Stairs-V04` |
| `PROTOCOL_ID` | `v04-stair-body-speed-v1` |
| 初始化 Actor | `logs/a2_ideal_stairs_v02_formal/monitor_counter_restart_20261112_200716/formal_seed42/chunk_0006_n512/checkpoints/budget_exhausted.pt` |
| 初始化 SHA256 | `2ab41ba8cdf18cf1079311af53f50d111e537546030cdf2ccaebcb333f73ff81` |
| 源版本/来源 schema | 0.2.1 / schema3；3000 更新、60000 Adam、512 环境、73728000 环境控制样本 |
| 奖励配置 SHA256 | `e2c2a134e7a2c9d7e66fd95cb7ce18e4db9cc149b5811994ac4741ae90ff225b` |

所有数值是 SDS 冻结设计值，不是经训练证明的最优值。本次没有调整网络结构、PPO 数学、速度指令、课程、随机化、动力学或终止条件。

## 2. 文件与行为

| 路径 | 作用 |
|---|---|
| `scripts/a2_v04_bootstrap.py` | 仅 stdlib 的 `src.tasks` namespace 隔离；阻止旧 `import_packages` 扫描；错 root/普通已加载 `src.tasks` 拒绝，同 root 幂等 |
| `src/tasks/velocity/v04/spec.py` | 版本/阶段/奖励参数/源码显式闭包/contract；source 收集排除 `src/tasks/__init__.py`，绑定 78 个产品源文件 |
| `src/tasks/velocity/v04/rewards.py` | 门控速度、姿态、非足、事件纯函数；`SwingClearanceState` 摆动目标/锁定/防重复领奖 |
| `src/tasks/velocity/v04/env.py` | `V04Env`：保持冻结物理/成功/失败/观测；替换奖励总贡献；摆动状态每控制步一次；足端停留诊断；env0 真实 5 ms 接触环形缓存 |
| `src/tasks/velocity/v04/model.py` | `V04ActorCritic`：结构/前向不变；只复制 0.2.1 Actor 全部参数/缓冲区和 std；Critic/统计/Adam 全新；另有 `from_zero` 接受当前 v04 zero 初始化容器（scope=initialization、空 Adam、零学习计数），不当作 resume |
| `src/tasks/velocity/v04/runner.py` | schema5 保存/恢复；课程/门禁；`reward_state='new_episode_no_half_trajectory'`；复用冻结 v03 PPO 数学 |
| `src/tasks/velocity/v04/recovery.py` | schema5 完整边界/账本/Adam/RNG 校验；拒绝 schema4 伪装、未知尾部、未保存更新、probe/formal 混用 |
| `src/tasks/velocity/v04/speed.py` | v0.3 速度统计/CPU oracle 数学逐字复用，仅版本身份导入调整 |
| `src/tasks/velocity/v04/evaluation.py` | v04 原始证据、案例/门禁、CPU float64 oracle、v04 奖励/摆动诊断 |
| `src/tasks/velocity/v04/connection.py` | 可靠发送最小复制；v04 group/readiness、`v04_heartbeat`；32 在途/2 s/30 s/120 s 不变；supervisor 在 child 等待循环内每 2 s 轮询 DeliveryPump，保证长 chunk 心跳/重试不只在 chunk 边界发生 |
| `src/tasks/velocity/v04/telemetry.py` | v04 run/group/面板；固定 8 组；swing/foothold 进入既有奖励/运动面板 |
| `src/tasks/velocity/v04/orchestration.py` | 唯一正式 supervisor；固定 512/8/42/cuda:0；原 gate/串行 GPU/单 writer；无多候选自动降档 |
| `scripts/check_a2_ideal_stairs_v04_migration.py` | CPU `--mode freeze/audit`；不动 CUDA/仿真/ResourceMonitor |
| `scripts/train_a2_ideal_stairs_v04.py` | 正式 worker；512-only，schema5，legacy manifest 校验，监控 swing/foothold |
| `scripts/run_a2_ideal_stairs_v04.py` | 正式 supervisor CLI 包装；SDS §9.3 固定参数 |
| `scripts/evaluate_a2_ideal_stairs_v04.py` / `play_a2_ideal_stairs_v04.py` | v04 评估/三速回放入口 |
| `scripts/smoke_a2_ideal_stairs_v04.py` | 独立 8 环境×2 更新冒烟；不使用正式 Curriculum/AttemptLedger/schema5 保存 |
| `scripts/test_a2_ideal_stairs_v04.py` | bootstrap 后运行 `tests/test_v04_*.py`，支持 `--pattern`，失败非 0 |
| `scripts/diagnostics/analyze_a2_v04_rewards.py` | 反例/收益排序/旧 trace 可重构项分析 |

## 3. 最终奖励公式与数值

记 `dt=0.02`，`[x]+=max(x,0)`，角度均为弧度。`c` 是本控制步实际执行的正命令。
`r_total` 由以下各项相加，**dt 只出现一次**；旧 `reward_progress` 置 0，仅诊断。

### 3.1 速度

```
ex = (vx - c) / c
ey = vy / 0.2
a  = 0.5 * (exp(-4*ex^2 - ey^2) - exp(-4))
p_over = 10 * max(vx - 1.05*c, 0)
p_back = 2  * max(-vx, 0)
p_side = 0.25 * tanh(ey^2)

r_speed = dt * ( g * max(a,0) + min(a,0) - p_over - p_back - p_side )
```

`g` 仅在以下全部成立时为 1：至少两足满足同面安全顶面支撑；`|roll|≤30°`；`-10°≤pitch≤55°`；本控制末采样无监测 link 合力模 `>10 N`；本控制步未失败；调用者启用运动奖励。门控只关闭指数正部，所有负项保留。`c=0` 的停车诊断显式 `enabled=False` 返回 0；活动 `c≤0`、非有限输入、shape 错、非法 `dt` 报错。
`vx=c,vy=0,g=1` 时每步约 `0.009816844`（即 `0.02×0.4908421806`），每秒 `0.490842/s`；`vx=1.35,c=0.2` 每秒约 `-11.40916`。

### 3.2 姿态/非足/事件

```
r_orientation = -dt * ( roll^2 + [pitch-55°]+^2 + 4*[-5°-pitch]+^2 )
r_nonfoot     = -dt * 1(any_j norm(link_force_j) > 10 N)
r_success     = +10 * (success_event & ~failure)
r_failure     = -4  * failure
```

非足严格 `>10 N`，等于 10 N 不命中；任一 link 命中就是 `-dt`，没有 1/9 稀释。成功/失败同帧只结算 `-4`。

### 3.3 落足与摆动

- 落足：冻结 `FirstFootfallState`，同足/同 tile/同阶连续两个合格控制样本后，每足每阶第一次 `+0.25`；失败同帧 0。
- 摆动：每足每目标阶/回合最多 `+0.10`；`r_swing=0.10*delta`，四足求和；不乘 dt；整回合上限 `≤0.4N`。
- 目标：`k=highest_safe_step+1`，仅 `1≤k≤N`；目标前缘距离 `0≤edge_x-foot_x≤0.30 m`；足横向在 tile 42 mm 内缩；目标顶面高于起跳支撑面。锁定后越沿/离开窗口不取消；窗口外起跳不锁定。
- 高度：`h_world=bottom_z_world-takeoff_bottom_z`；`h_body=foot_body_z-takeoff_body_z`；`H=target_top_z-takeoff_support_z+0.04`；`phi=clamp(min(h_world/H,h_body/0.04),0,1)`；`delta=max(phi-paid_fraction,0)`。
- `paid_fraction` 在锁定目标后无论门控/失败/接触都消费；`actual_paid_amount` 只在真实支付时增加；重复尝试同阶不重置，只有 episode reset 清除；失败/门控恢复不能补领。
- 足底/机身坐标：`foot_pos_b = quat_apply_inverse(root_quat_w, foot_pos_w - root_pos_w)`，先平移再逆旋转。
- 初始化检查足球：每腿唯一 `*_foot_collision`、球形、半径 0.032 m（1e-6）、与 foot site 同 body 且球心/site 重合（1e-6）；不匹配拒绝运行。

### 3.4 未改连续项

`track_angular_velocity`（权重 1，活动减基线 1）、`pose`（0.2，减基线 1）、`body_ang_vel` −0.05、`angular_momentum` −0.025、`joint_acc` −2.5e-7、`joint_limits` −10、`action_rate` −0.05、`slip` −0.25、`soft_landing` −0.001、`stand_still` −1、`center` −0.5、`heading` −0.5、`edge` −1。旧机身 x 进度收益删除，实际总贡献恒 0，仅 `diagnostic/old_body_progress` 供比较。

## 4. 训练/评估关键设置

| 设置 | 值 |
|---|---|
| 训练/正式评估环境数 | 512 / 8（v04 固定；无候选数组自动降档） |
| seed / device | 42 / `cuda:0` |
| 物理 dt / decimation / 控制 dt | 0.005 / 4 / 0.02 |
| rollout | 48 控制步；每更新 24576 环境控制样本 |
| PPO | 5 epochs × 4 minibatches = 20 Adam/更新；初始 lr 1e-4；范围 [1e-5,3e-4]；desired KL 0.01 |
| clip / gamma / GAE λ | 0.2 / 0.99 / 0.95 |
| entropy / value coef / max grad | 0.005 / 1.0 / 1.0 |
| std 范围 | [0.05,0.30]；迁移原 Actor std |
| 保存/阶段评估 | 每 100 / 每 200 更新 |
| 指令 | 每回合 U[0.2,0.6] m/s，回合内固定；vy=0，wz=0 |
| 总预算 | 23000 更新；S1 3000/S2 4000/S3 4000/S4 5000/S5 7000 |
| 阶段槽位 | S1 全部当前；之后前阶段 20%、当前 80%（512 时 102/410） |
| 联合成功 | 原 safe-arrival + 原速度门禁；每 200 更新三速各 20 回合，每档 ≥18/20，连续两次；S1 额外扰动；S5 第二次后冻结，test seed400342 各 100 回合每档 ≥95/100 |
| 正式证据计数 | schema5 formal checkpoint 必须 512；正式 validation/perturbed/test/baseline 必须 actual_num_envs=8；replay 1；probe 8/4/1 仅后续资源诊断；无候选降档 |

## 5. 已执行命令与结果

### 5.1 V0 来源冻结

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
CUDA_VISIBLE_DEVICES= /home/xxs/anaconda3/envs/a2/bin/python \
  scripts/check_a2_ideal_stairs_v04_migration.py --mode freeze --device cpu \
  --project-root /home/xxs/dog/unitree_rl_mjlab_a2 \
  --source-checkpoint logs/a2_ideal_stairs_v02_formal/monitor_counter_restart_20261112_200716/formal_seed42/chunk_0006_n512/checkpoints/budget_exhausted.pt \
  --source-checkpoint-sha256 2ab41ba8cdf18cf1079311af53f50d111e537546030cdf2ccaebcb333f73ff81 \
  --legacy-source-root /home/xxs/dog/unitree_rl_mjlab_a2 \
  --v03-replay-checkpoint logs/a2_ideal_stairs_v03/authorized_fork_2700_20261113_173259/checkpoints/monitoring_paused.pt \
  --v03-replay-sha256 a316c649af43ebb76a12b7f2fba429aca3d0da465996706297c039ce9b12eeef \
  --v03-replay-source-root /home/xxs/dog/unitree_rl_mjlab_a2 \
  --output-dir logs/a2_ideal_stairs_v04/implementation/legacy_snapshot
```

结果：Actor 源 43 项源码全部哈希匹配；v0.3 replay 132 项源码单独冻结；未初始化 CUDA。证据：`logs/a2_ideal_stairs_v04/implementation/legacy_snapshot/`。

### 5.2 迁移审计（CPU）

```bash
CUDA_VISIBLE_DEVICES= /home/xxs/anaconda3/envs/a2/bin/python \
  scripts/check_a2_ideal_stairs_v04_migration.py --mode audit --device cpu \
  --project-root /home/xxs/dog/unitree_rl_mjlab_a2 \
  --source-checkpoint logs/a2_ideal_stairs_v02_formal/monitor_counter_restart_20261112_200716/formal_seed42/chunk_0006_n512/checkpoints/budget_exhausted.pt \
  --source-checkpoint-sha256 2ab41ba8cdf18cf1079311af53f50d111e537546030cdf2ccaebcb333f73ff81 \
  --legacy-source-manifest logs/a2_ideal_stairs_v04/implementation/legacy_snapshot/legacy_manifest.json \
  --output-dir logs/a2_ideal_stairs_v04/implementation/migration_audit
```

结果：`status=passed`，27 个 Actor/std tensor 全等，128 组 CPU 观测 Actor 均值 `max_abs=0.0`，fresh Critic、空 Adam、零学习计数，`cuda_initialized=false`、`imported_src_tasks=true`（实际测量）。证据：`migration_audit/migration_audit.json`、`migration_zero.pt`、`source_inventory.json`；最终 zero SHA256 `8d3f4c13bf71d41e90764791fb7cbeef0f5e78a2db848e0a554db4b1cc1e0423`。

**protocol_id 修复后的复测（2026-11-15）**：同一命令、输出改到 `migration_audit_protocol_fix/`，`status=passed`，`max_abs=0.0`、`cpu_actor_mean.allclose=True`、`fresh_adam_empty=true`、`cuda_initialized=false`；zero SHA256 随 `env.py` 源码/行为哈希变化而更新为 `2e5bf37cec3be7148c2b226a858e4dbcdcaa4f46728dded5edd190a5f387c4e3`。旧 zero SHA 只对应修复前源码，不得再引用为当前源码证据。

### 5.3 CPU 测试

```bash
CUDA_VISIBLE_DEVICES= /home/xxs/anaconda3/envs/a2/bin/python \
  scripts/test_a2_ideal_stairs_v04.py --pattern 'test_v04_*.py'
```

分模式证据：`logs/a2_ideal_stairs_v04/implementation/cpu_tests/`。

- 非评估套件（14 个 `test_v04_*.py` 模块）：**216 项全部通过**，92.3 s，exit 0（`non_evaluation_final.log`）。
- 评估套件：**37 项全部通过**，302.1 s，exit 0（`evaluation_final.log`；运行时间较长由 60/300 案例 fixture 与 CPU 竞争造成，不是挂起）。
- 合计：**253 项 CPU 测试通过**；官方入口 `--pattern 'test_v04_*.py'` 最终全量运行 `Ran 253 tests in 374.587s, OK`，exit 0（`all_v04_final.log`）。测试入口在 bootstrap 后 discover；不得在已 bootstrap 进程混跑旧 v03 测试。

**protocol_id 修复后的复测（2026-11-15）**：官方入口全量运行 `Ran 253 tests in 364.763s, OK`，exit 0（`cpu_tests/all_v04_protocol_fix.log`）。这是当前源码（含 `V04Env.step` 身份改写）对应的有效 CPU 证据；此前 374.587 s 的日志早于该修复，不再作为当前源码证明。

### 5.4 8 环境×2 更新 GPU 冒烟

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
MUJOCO_GL=egl /home/xxs/anaconda3/envs/a2/bin/python \
  scripts/smoke_a2_ideal_stairs_v04.py \
  --task Unitree-A2-IdealHeightMap-Stairs-V04 --stage S1 --num-envs 8 --updates 2 \
  --seed 42 --device cuda:0 --vx-min 0.2 --vx-max 0.6 --vy 0.0 --wz 0.0 \
  --init-checkpoint logs/a2_ideal_stairs_v02_formal/monitor_counter_restart_20261112_200716/formal_seed42/chunk_0006_n512/checkpoints/budget_exhausted.pt \
  --init-checkpoint-sha256 2ab41ba8cdf18cf1079311af53f50d111e537546030cdf2ccaebcb333f73ff81 \
  --legacy-source-manifest logs/a2_ideal_stairs_v04/implementation/legacy_snapshot/legacy_manifest.json \
  --output-dir logs/a2_ideal_stairs_v04/implementation/smoke_gpu
```

结果（实际 GPU）：`status=passed`，2 PPO / 40 Adam / 96 控制步 / 768 环境控制样本；finite reward/obs/loss/grad 通过；96/96 reward 总和，最大和误差约 `9.54e-7`；`reward_progress` 非零 0；`swing`/`foot_dwell`/`reward_gate` 均存在；`smoke_state.pt` 容器 `v04-smoke-state-v1`，SHA256 `9c65507052ce274a1e6c30bd8adbe3bc6d658959b436471acd7feb8e1cff1680`，正式 preload 拒绝。证据：`smoke_gpu/result.json`、`smoke_gpu/smoke_state.pt`、`smoke_gpu/stdout_stderr.log`。此结果只证明工程接线/有限性，不证明爬楼。

**protocol_id 修复后的复测（2026-11-15）**：同一命令、输出 `smoke_gpu_protocol_fix/`，`status=passed`、`gpu_acceptance=true`、`formal_resumable=false`；2 PPO / 40 Adam / 96 控制步 / 768 样本；`reward.max_sum_abs_error=4.76837158203125e-07`；`smoke_state` SHA256 `1aef5f1c9d28b96404175c7174fc990991acb44df08d0d7525d13684457934bd`，正式 preload 以 `Strict schema5 checkpoint field contract mismatch` 拒绝；耗时 17.7 s。旧 `9c6550…` 只对应修复前源码。

### 5.5 反例与收益排序

```bash
CUDA_VISIBLE_DEVICES= /home/xxs/anaconda3/envs/a2/bin/python \
  scripts/diagnostics/analyze_a2_v04_rewards.py \
  --trace-dir logs/a2_ideal_stairs_v03/view_model2721_20261113_190230
```

结果：A/B/C 45 个 `(variant,c,N)` 单元、225 条合成轨迹、135 个必需排序比较全部通过（容差 1e-6）；反例通过；旧 trace 只重算可重构项，`full_v04_return_reconstructable=false`。证据：`logs/a2_ideal_stairs_v04/implementation/reward_analysis/report.json`（SHA256 见同目录 `report.sha256`）。合成轨迹不代表动力学可行或训练效果。

### 5.6 正式 supervisor 预检（不启动训练）

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
/home/xxs/anaconda3/envs/a2/bin/python scripts/run_a2_ideal_stairs_v04.py \
  --task Unitree-A2-IdealHeightMap-Stairs-V04 --recipe-version 0.4-r1 \
  --init-checkpoint logs/a2_ideal_stairs_v02_formal/monitor_counter_restart_20261112_200716/formal_seed42/chunk_0006_n512/checkpoints/budget_exhausted.pt \
  --init-checkpoint-sha256 2ab41ba8cdf18cf1079311af53f50d111e537546030cdf2ccaebcb333f73ff81 \
  --legacy-source-manifest logs/a2_ideal_stairs_v04/implementation/legacy_snapshot/legacy_manifest.json \
  --num-envs 512 --eval-num-envs 8 --seed 42 --device cuda:0 \
  --vx-min 0.2 --vx-max 0.6 --vy 0.0 --wz 0.0 \
  --max-iterations 23000 --worker-chunk-iterations 200 \
  --save-interval 100 --eval-interval 200 \
  --wandb-mode online --wandb-project <沿用项目配置> --wandb-entity <沿用账户配置> \
  --wandb-run-name <独立v04运行名> \
  --output-dir <不存在的v04输出目录> --stop-after-preflight
```

本次仅验证静态预检，不创建可恢复零更新 checkpoint、不启动 SDK/训练。

### 5.7 正式训练实际执行（2026-11-14 23:43 起）

- 命令：`scripts/run_a2_ideal_stairs_v04.py`，512 训练 env、8 评估 env、seed 42、cuda:0、vx 0.2–0.6、vy=wz=0、23000 总预算、200 更新/chunk、save 100、eval 200、W&B online；初始化 checkpoint 为 v0.2.1 批准 SHA `2ab41ba8…ff81`，输出 `logs/a2_ideal_stairs_v04/formal_seed42_20261114_234305`。原启动参数与 W&B 运行名记录在 `logs/a2_ideal_stairs_v04/current_launch.env`。
- 首次启动（`formal_seed42_20261114_234114`）在 0 更新 baseline 阶段因 `protocol_id` 溯源不一致失败；修复 `V04Env.step` 后于 23:43 重启，本次为重启后的正式运行。
- 实际门禁：200/400/600/800/1000/1200/1400/1600/1800/2000 共 10 次 S1 joint validation，三速（0.2/0.4/0.6 m/s）均 **0/20**、`climb_successes=0`、`joint_successes=0`，失败原因全部为 `timeout`；从未晋级，`consecutive_passes=0`。
- 行为：第 1 次更新 `actual_vx_body=1.2435 m/s`（指令 0.377），第 150 次更新降至 0.0062，之后长期约 0；900–1200 更新均值约 0.002–0.007 m/s。训练在约 100–150 更新内把冻结 Actor 的步态压成站立。
- 奖励归因（`0.4-r1` 原样执行）：10–40 更新段 `reward_total=-0.291/步`（速度项 -0.1717，超速罚 `10*(v-1.05c)` 主导），站立段 `reward_total≈-0.004/步`（速度项≈0）；`speed_reward` 在 v=0 时指数项与 `over/back` 恰为 0，站立即零罚稳态。即 0/20 的根因是初始化 Actor 天然速度（约 1.24 m/s）远离 0.2–0.6 指令带叠加超速重罚，属于配方/初始化匹配问题，不是代码崩溃或证据缺陷。
- 终止情况（最终）：2000 门禁（2026-11-15 06:06）后 W&B 投递连续 `RemoteReadError`/`Service Unavailable`/SSL EOF，超过 120 s 上限触发 fail-closed `monitoring_blocked`，supervisor 退出且未写 `final.json`（暂停在 2000/23000，checkpoint `chunk_0009_n512/checkpoints/latest.pt` SHA256 `943ff7e1a45a6b12db8cc25ee0466eca7a17a6e368cbb1c9333ea76a494832bc`）。
- 三次 resume 尝试与最终终止：resume1 在 06:20 恢复后再次被 fail-closed pause，worker 写出完整 checkpoint `chunk_0010_n512/checkpoints/monitoring_blocked.pt`（SHA256 `b2f0018b7b97789ae0e58db2c82ef516ca54157550ead2dbf34268a03140e76e`，2000 更新、40000 Adam step）；resume2 在 spawn 前单核空转挂起被终止；resume3 先把 abort 的 `chunk_0010_n512` 及其 process/control/log 改名为 `*_aborted_monitoring_20261115_0632` 保留，再以 state 指定的 `943ff7e1…` 续跑，worker 审计拒绝：`newer complete checkpoint available; old same-update state forbidden`（resume1 的 `monitoring_blocked.pt` 已成为更新的完整 checkpoint，旧同更新状态不得再用），supervisor 写 `final.json` 进入**终端** `status=recovery_blocked`。
- 终端证据：`final.json`（iteration=2000、env_steps=49152000、`accounting.completed_ppo_updates=2000`、`discarded_samples=0`、W&B `wandb_finish.status=remote_verified`）；`supervisor_state.json` phase=terminal/status=recovery_blocked，curriculum 为 S1/stage_iter=2000/consumed.S1=2000/history 9 条。注意 `final.json` 内嵌 `result.curriculum` 是失败子进程返回的空 curriculum，与 state 不一致，属终端 payload 展示问题，权威计数以 state 与 accounting 为准。
- 该运行现已 terminal，`--resume-run` 对终端状态直接拒绝；`approved_init_kind` 只接受冻结 v0.2.1 Actor（`2ab41ba8…`）或当前 contract 下的零更新容器，2000 边界完整 checkpoint（`b2f0018b…`）不能作为新运行的 init。因此继续训练只有两条路：① 以 v0.2.1 初始化**从 0 新开**（配方不变，重跑约 2000+ 更新，预计仍是站立稳态）；② 用户授权修改恢复/监控代码或修改配方后新开；两者均需用户决定。
- 未改奖励/课程/速度/终止/PPO/监控阈值；未 commit/push。W&B：`https://forge.coreweave.com/wandb/xingshenxu-/a2-ideal-stairs-v02-formal/runs/058510996fe8461390811ae9afb3c1aa`。

## 6. 后续正式运行命令模板

以下模板用于**新开**一次正式运行；本次正式运行的实际参数与结果见 5.7，输出目录必须不存在。

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
/home/xxs/anaconda3/envs/a2/bin/python scripts/run_a2_ideal_stairs_v04.py \
  --task Unitree-A2-IdealHeightMap-Stairs-V04 --recipe-version 0.4-r1 \
  --init-checkpoint logs/a2_ideal_stairs_v02_formal/monitor_counter_restart_20261112_200716/formal_seed42/chunk_0006_n512/checkpoints/budget_exhausted.pt \
  --init-checkpoint-sha256 2ab41ba8cdf18cf1079311af53f50d111e537546030cdf2ccaebcb333f73ff81 \
  --legacy-source-manifest logs/a2_ideal_stairs_v04/implementation/legacy_snapshot/legacy_manifest.json \
  --num-envs 512 --eval-num-envs 8 --seed 42 --device cuda:0 \
  --vx-min 0.2 --vx-max 0.6 --vy 0.0 --wz 0.0 \
  --max-iterations 23000 --worker-chunk-iterations 200 \
  --save-interval 100 --eval-interval 200 \
  --wandb-mode online --wandb-project <沿用项目配置> --wandb-entity <沿用账户配置> \
  --wandb-run-name <独立v04运行名> \
  --output-dir <不存在的v04输出目录>
```

续训使用互斥的 `--resume-run --resume <当前 supervisor checkpoint> --resume-sha256 <SHA>`，不得与 init 同时出现；实测该路径可被 W&B fail-closed 或 supervisor 启动挂起打断，恢复前需先确认监控可用。

## 7. 未完成/未验证

- 正式训练已实际启动，跑到 2000/23000 更新后因 W&B fail-closed 与恢复策略而**终端终止**（`status=recovery_blocked`，见 5.7）；未完成 S1 3000 预算终点、S2–S5 课程与最终 100 回合×三速验收，当前无任何爬楼成功证据。
- 512 环境资源生命周期 probe 尚未单独执行。
- supervisor `--resume-run` 路径实测暴露三个问题：① abort 的 chunk 目录会与 worker `mkdir(exist_ok=False)` 冲突（resume3 已用改名规避）；② resume2 在 spawn worker 前单核 ~100% CPU 空转，根因未定位（W&B SDK 同期正常，每 15 s `200 OK`）；③ 跑出更新的完整 checkpoint 后，旧同更新状态被恢复审计正确拒绝，但 CLI 要求 resume 必须使用 state 里较旧的 checkpoint，导致无法自愈。
- 8×2 GPU 冒烟只证明工程接线/有限性，不证明策略学会爬楼、成功率或奖励改善。
- 合成轨迹/单测/反例排序不是动力学可行性或训练效果证明。
- 在线 W&B 在本次正式运行中确实建立了 8 面板并收到事件（`remote_verified`），但投递可靠性不足以支撑长时间训练：2000 门禁后出现 `RemoteReadError`/SSL EOF 并使运行 fail-closed 暂停；正式多 seed 稳健性仍未验证。
- v04 正式多候选环境自动降档入口已删除，contract 的 resource candidate 已固定 `[512]`/`[8]`；正式 checkpoint preload/save 要求 512，正式评估证据要求 training=512/actual=8；`ProbeCurriculum` 仅保留给后续 512 资源 probe。`runner.py` 仍保留 legacy `ENV_COUNTS` 以兼容 probe 路径，但 formal 路径会拒绝非 512。
- 独立代码审核（两轮）发现的真实问题已修复：正式 gate 的 actual_num_envs=8、schema5 环境数一致、supervisor init SHA/zero init、配置 offset 证据、DeliveryPump 在 child 生命周期内 2 s 轮询与 `v04_heartbeat`、audit/smoke 完整性与实际 flags、worker 参数落盘。两轮审核均为只读；主 agent 核验后修复并重跑。

## 8. 证据索引

- V0 冻结：`logs/a2_ideal_stairs_v04/implementation/legacy_snapshot/`
- CPU 迁移审计：`logs/a2_ideal_stairs_v04/implementation/migration_audit/`；protocol_id 修复后复测 `migration_audit_protocol_fix/`
- CPU 测试：`logs/a2_ideal_stairs_v04/implementation/cpu_tests/`（当前源码证据 `all_v04_protocol_fix.log`）
- GPU 冒烟：`logs/a2_ideal_stairs_v04/implementation/smoke_gpu/`；protocol_id 修复后复测 `smoke_gpu_protocol_fix/`
- 正式训练运行：`logs/a2_ideal_stairs_v04/formal_seed42_20261114_234305/`（`supervisor_state.json`、`formal_ledger/`、`validation_*/summary.json`、`chunk_*/checkpoints/`）；启动记录 `logs/a2_ideal_stairs_v04/current_launch.env`
- 奖励排序/反例：`logs/a2_ideal_stairs_v04/implementation/reward_analysis/`
- 独立审核记录：`logs/a2_ideal_stairs_v04/implementation/agent_notes/independent_review_fixes.md`、各 agent notes 与 `doc/sds/reviews/`
- 修改日志：`logs/codex_changes.md`
