# S2.1 训练执行栈（T12，CPU/fake 已完成，GPU 待开窗）

## 角色与课程

- 1024 env 角色分配：205 / 205 / 410 / 102 / 102 = flat_walk / flat_stop /
  stairs_walk / platform_walk / l40_milestone（20/20/40/10/10，最大余数法，
  固定角色顺序）。
- `stairs_walk`：10/20 级各 205；每级内 0.8/4.0 m 宽 × 0.3/0.5/0.6 m/s
  六组轮转平衡；登顶后固定 5 m。
- `platform_walk`：2/4 m 顶高各 51；宽 0.8/4.0 × 三速轮转平衡。
- flat_walk / flat_stop / L40 各自速度/停车方式组内轮转。

## PPO / 损失

- 48 步 rollout；1024×48 = 49,152 样本/rollout。
- 5 epoch × 4 逻辑 minibatch；micro 512；Adam 每逻辑 minibatch 一次，
  先按全逻辑批 KL 调 LR，再一次 Adam。
- LR 3e-5，范围 [1e-5,1e-4]；std 从入口解冻，投影 [0.05,1]；
  `S2.1` 新建 Adam；教师 eval/no-grad、normalizer 冻结；课程切换不重置
  LR/std。
- FP32、grad clip 1.0、entropy/approx KL、按样本数加权累积。
- `s2_loss.AnchorAccumulator` 提供 `n_clean` 分母的 `L_anchor`；CPU 测试
  覆盖分母与总损失。

## 预算

- 每 rollout `learning_S2_1` reserve 49,152 → commit actual → release unused。
- S2.1 完整 rollout：`floor(800,000,000 / 49,152) = 16,276`；
  checkpoint 每 5,000,000 样本（102 rollouts）。
- 未写台账；CLI 只在给定 ledger/预算存在时运行。

## 检查点

`s2_checkpoint` schema 11；`S2Trainer.checkpoint_metadata` 写：
course/curriculum/strength、model/optimizer、policy_std、normalizer、
teacher/entry、beta_flat、learning/eval/engineering/rollback samples、
gates_passed/gate_ab_sequence/gate_uses/final_uses、case/candidate/code/
manifest SHA、 ledger_run_id/ledger_cursor、random_state、role_assignment、
randomization/command_flow/delay_config。resume 恢复完整模型/优化器、
LR、std、rollouts 与预算游标，不重置。

## 门禁与资源

- S2.1 gate matrix 36 组（flat 9 + stairs 12 + platform 12 + L40 3），
  每组 20 例、A/B 两套、阈值 18/20；L40 使用
  `l40_milestone_cache_gate_A/B`。
- `scripts/probe_a2_v09_s2_resources.py --plan-only` 输出 §2.3 全项计划；
  `--execute` 需真实 1024-env 训练装配，当前不伪装 fake 为 GPU 准入。
- 真实 probe：5 warmup + 50 measurement + restore，完整 48 步/20 Adam；
  峰值上限 7168 MiB、host free ≥4 GiB；失败写证据，再按 512。
- `scripts/train_a2_v09_s2.py --execute` 需要 reserve、probe admitted、
  independent admission；CPU 栈测试用 `--fake-backend`，真实 GPU 装配
  尚未安装，不会静默用 fake 训练。

## GPU 窗口估算（provisional）

```text
probe 1024 control samples: 3,709,184
training full:             16,276 rollouts × 49,152 = 800,000,000
```

实际显存/时长必须由 probe 实测；单环境实测约 1.5 GiB 不能外推到 1024。
预留窗口按“probe 1–2 h + 训练 10–20 h + 门禁/检查点余量”申请，
最终以 probe 报告为准。

## T13 真实装配与 probe（2026-11-22）

- `src/tasks/velocity/v09/s2_train_real.py`：
  - `build_env_plan(1024)` 角色→S2 terrain/tile/spawn；
  - `S2TrainEnv` 基于 `V07Env`，注入 S2 terrain patch 与 40×30
    `v09.map.HeightMap`；每 env 按角色 tile 分配；
  - 观测合同：`proprio 45 / history 5×45 / heightmap 3×40×30 / route_error 3 /
    privileged 35 / Critic 211`；
  - `assert_v07_contract` 校验 control dt / physics dt / decimation /
    action scale / joints / map grid/refresh。
- `S2TrainEnv` 的物理构造需要真实 mjlab/warp；CPU 测试只校验静态计划与合同，
  不构造真实物理。
- `scripts/probe_a2_v09_s2_resources.py --execute` 已接 `S2TrainEnv` + `S2Trainer`
  的 5 warmup + 50 measured rollout + restore；当前 probe 报告仍不含
  evaluation/video 段（`evaluation_envs=0`），因此**不能**把它当作完整 §2.3
  admission；该段需在 GPU 窗口补齐后再启动训练。
- `train --execute` 仍要求 probe admitted + independent admission；CPU 用
  `--fake-backend`。

### 最可能 GPU 首爆点

1. 1024 env 的 terrain/map 更新与 `V07Env` 的 `origins/tile_ids` 同步：首轮
   reset 在地形 patch 外或 tile 0/自定义 tile 交错处最容易 OOB/错 tile；
2. 1024×3×40×30 heightmap + history 的显存放大；
3. PPO 逻辑批 49,152 的 heightmap 拼接（~700 MB）与 5×4 micro 的峰值；
4. probe 的 evaluation/video 段尚未接入，不能提前报 admitted。
