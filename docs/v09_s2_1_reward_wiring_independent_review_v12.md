# V12 独立复核：S2.1-r1 奖励接线、作废边界与修复候选

- 复核角色：A2-S2-VERIFY（独立于实现方）
- 复核方式：**纯 CPU、只读**；未运行 GPU/物理/训练，未修改实现、测试、SDS、证据、ledger、checkpoint，未做 git 写入。
- 审计快照：`HEAD=1ec58f8`（`feat(s2): vectorized S2 reward adapter and audit tests`，2026-11-22 23:46:49 +0800）；其父提交 `a744090` 是 r1 作废与修复前文档提交。
- r1 实现基线：`a744090` 的 `src/tasks/velocity/v09/s2_train_real.py` 与最后改动该文件的 `3f41582` 均无 `_reward` 覆写；`3f41582` 的文件 SHA256 为 `0d4a36fd5dbc094aea928bd30a06ed22120526cda35f3059c2a06a2c86f12fb4`。
- SDS：`doc/sds/SDS_A2_v0.9_Sequential_Map_Walk_Stop_Robustness.md` SHA256 `09fa2d4cdd858914c46a51fd52670c20cb20f2b6f3e76e6d266054574f64ccbe`。
- 结论：**r1 缺陷确认；四个 r1 检查点全部作废；重跑不批准**。当前 `1ec58f8` 已把训练奖励接到 S2 批量实现，但仍存在多项 §5 语义阻断，**不能作为 s2.1-r2 的准入依据**。

## 0. 结论摘要（逐项确认/否认）

| 核查项 | 结论 |
|---|---|
| r1 的 `S2TrainEnv` 没有覆写 `_reward`，训练由 `V07Env._reward → reward_computer` 产生奖励 | **确认** |
| r1 的训练路径调用 `s2_rewards.compute_reward` | **否认** |
| r1 的四个 S2.1 检查点可用于门禁/终验/选模 | **否认**（全部作废） |
| S2.0 诊断/报告因本次奖励缺陷被污染 | **否认**（推理链路不使用奖励） |
| 台账 `learning_S2_1 spent=20,275,200 / 800,000,000`、`reserved=0`，属失败消耗保留 | **确认** |
| `run_a2_v09_s2_gate.py` 会在代码层拒绝这四个作废 SHA | **否认**（仅文档声明，代码未拒绝） |
| 当前 `1ec58f8` 已把训练奖励改为 S2 实现 | **确认**（接的是 `compute_reward_batch`，不是标量 `compute_reward`） |
| 当前 `1ec58f8` 满足 §5、可批准重跑 | **否认**（见 §5，存在 P0 阻断） |

## 1. r1 的奖励调用链（确认：实际使用 v07 17 项）

### 1.1 代码链

在 r1 的基线 `a744090`：

1. `S2TrainEnv` 不是继承 `V07Env`，而是**组合**：`self.env = V07Env(...)`（`src/tasks/velocity/v09/s2_train_real.py:168-172`，`a744090`）。因此“继承”这一措辞不精确，但结论不受影响。
2. `S2TrainEnv` 只覆写 `reset/step/close`，`step` 直接委托 `self.env.step`：

```python
# a744090:src/tasks/velocity/v09/s2_train_real.py:197-199
def step(self, actions):
    obs, reward, terminated, truncated, info = self.env.step(actions)
    return obs, reward, terminated | truncated, info
```

3. `S2TrainEnv.__getattr__` 也只做属性委托（同文件 `190-191`）；该类没有 `_reward` 方法。
4. 该文件在 `a744090` 的导入行只有 `s2_physics, s2_spec, s2_terrain, s2_train_env`（同文件 `11`），**没有导入 `s2_rewards`**。
5. `V07Env.__init__` 构造 v07 奖励计算机：

```python
# src/tasks/velocity/v07/env.py:294
self.reward_computer = reward_module.RewardComputer()
```

6. `V07Env._reward` 调用它，并把 `weighted_terms/total` 写回：

```python
# src/tasks/velocity/v07/env.py:709-712
def _reward(self, state, previous_action):
    frame = self._reward_frame(state, previous_action)
    result = self.reward_computer(frame)
    total = result.total
```

7. `V07Env.step` 在 `:770` 调用 `reward = self._reward(state, previous)`，随后把该 `reward` 返回给 `S2Trainer`。
8. `S2Trainer.collect_rollout` 直接消费环境返回的 `reward`，没有第二次奖励计算：

```python
# src/tasks/velocity/v09/s2_train.py:139
next_obs, reward, done, info = self.env.step(actions.detach())
...
# :145
rewards = torch.stack([row['reward'] for row in transitions], 1)
```

`scripts/train_a2_v09_s2.py:119-126` 构造的正是 `s2_train_real.S2TrainEnv` 与 `S2Trainer`；没有奖励覆盖参数。

### 1.2 静态检索证据

在 `a744090` 上执行：

```text
git grep -n "_reward\|reward_computer" a744090 -- \
  src/tasks/velocity/v09/s2_train_real.py \
  src/tasks/velocity/v09/s2_train.py \
  scripts/train_a2_v09_s2.py
```

结果：**无匹配**（退出码 1）。

```text
git grep -n "s2_rewards\|compute_reward" a744090 -- '*.py'
```

结果只有：

- `src/tasks/velocity/v09/s2_rewards.py:137: def compute_reward(frame)`（定义本身）；
- `tests/test_v09_s2_0_rewards.py` 的测试调用。

没有任何训练文件导入或调用 `s2_rewards.compute_reward`。

### 1.3 时间边界

- 最后一个 r1 检查点 `s2_1_000020054016.pt` 的 mtime 为 **2026-11-22 23:40**。
- 修复提交 `1ec58f8` 的时间为 **2026-11-22 23:46:49**，晚于最后一个检查点，也晚于停训。
- 因此 r1 的四个检查点在物理上不可能使用该修复；它们对应的训练代码是 `1ec58f8` 之前的版本（`a744090`/`3f41582` 链路）。

### 1.4 运行日志旁证

`logs/a2_v09/s2/s2_1.log:97`：

```text
[INFO] <RewardManager> contains 0 active terms.
```

这说明训练没有使用 mdp `RewardManager` 的项；奖励来自 `V07Env` 的自定义 `RewardComputer`，与静态代码链一致。该日志不能单独证明“没有 S2 奖励”，但与代码链、提交时间边界共同构成完整证据。

**判定：确认 r1 使用 v07 17 项奖励；否认 r1 使用 SDS §5 的 S2 奖励。**

## 2. v07 17 项与 SDS §5.3 的 20 项逐项差异

SDS §5.3 表在 `doc/sds/SDS_A2_v0.9_Sequential_Map_Walk_Stop_Robustness.md:619-640`；v07 的 17 项在 `src/tasks/velocity/v07/spec.py:331-355`。

| # | SDS §5.3 项 | 巡航权重 | 减速/停车权重 | v07-r1 的实际状态 | 差异 |
|---:|---|---:|---:|---|---|
| 1 | 前进速度跟踪 | +1.5 | +1.5 | 同式同权 | 无 |
| 2 | 偏航角速度跟踪 | +0.8 | +0.8 | 同式同权 | 无 |
| 3 | 竖直速度 | -1.0 | -1.0 | 同式同权 | 无 |
| 4 | roll 姿态 | -5.0 | -5.0 | 同式同权 | 无 |
| 5 | 水平面 pitch `g_level·gx²` | **-5.0** | **-5.0** | **无此项** | **缺项** |
| 6 | 机身高度 | -0.1 | -0.1 | 同式同权（`base_height_above_surface`） | 无 |
| 7 | 动作变化 | -0.01 | -0.01 | 同式同权 | 无 |
| 8 | 默认关节姿态偏差 | -0.05 | -0.05 | 同式同权 | 无 |
| 9 | 关节加速度 | -2.5e-7 | -2.5e-7 | 同式同权 | 无 |
| 10 | 关节速度 | -5e-4 | -5e-4 | 同式同权 | 无 |
| 11 | roll/pitch 角速度 | -0.05 | **-0.20** | **-0.05 / -0.05** | **减速列缺 -0.20** |
| 12 | 足部腾空时间 | +0.2 | **0** | **+0.2 / +0.2**，停车不清累计 | **减速列与清零缺失** |
| 13 | 足部滑移 | -0.15 | -0.15 | 同式同权 | 无 |
| 14 | 抬脚高度误差 | -0.5 | -0.5 | 权重同；目标按 **tile** 选，楼梯 tile 的平台仍用 0.22 | **逐足真实表面缺失** |
| 15 | 关节跟踪 | -0.1 | -0.1 | 同式同权 | 无 |
| 16 | 横向居中 | -0.5 | -0.5 | 同式同权 | 无 |
| 17 | 朝向 | -0.5 | -0.5 | 同式同权 | 无 |
| 18 | 软关节越限 | -10.0 | -10.0 | 同式同权 | 无 |
| 19 | 水平面超速 `g_level·max(...)²` | **-2.0** | **-2.0** | **无此项** | **缺项** |
| 20 | 稳定站立 `I_stable` | 0 | **+1.0** | **无此项** | **缺项** |

### 2.1 三个缺项

v07 的 `REWARD_TERMS` 只有 17 个（`v07/spec.py:331-336`），缺少：

1. `orientation_pitch_level`（SDS 第 5 项，`-5·g_level·gx²`）；
2. `horizontal_overspeed`（SDS 第 19 项，`-2·g_level·max(v_heading-c_exec-0.15,0)²`）；
3. `stable_stand`（SDS 第 20 项，仅零指令稳定站立 `+1`）。

v07 的 `RewardComputer.__call__` 的 raw 项也只构造这 17 项（`v07/rewards.py:673-695`），不存在“项在但权重为 0”的情况。

### 2.2 一次性项与停车语义

| 项 | SDS §5.1 | v07-r1 | 证据 |
|---|---|---|---|
| 登顶 | +2，一次，仅登顶事件 | **没有** | v07 仅 `success/failure`，见 `v07/rewards.py:703-711` |
| 成功 | +8 | **+10** | `v07/spec.py:357`、`v07/rewards.py:703-707` |
| 失败 | -10 | -10 | `v07/spec.py:358`、`v07/rewards.py:705-707` |
| 超时 | 按失败 -10 | **0** | `v07/spec.py:359`；`_reward_frame` 只传 `success/failure`（`v07/env.py:704-705`），`RewardComputer` 只读 `frame.failure`（`v07/rewards.py:705`） |
| 前进位移 | 仅巡航 `0.4·Δx·I_cruise` | **每步 `0.4·Δx`** | `v07/rewards.py:699-702` 无 cruise 门控 |
| 减速/保持列 | `DECEL/HOLD` 两列 | **不存在** | v07 无 phase 列；`RewardComputer` 用固定权重 `v07/rewards.py:696-698` |
| 停车关闭腾空 | 关闭并清累计 | **不关闭、不清零** | `v07/rewards.py:685,696-698`；`AirTimeState` 无 stop 输入 |

### 2.3 抬脚目标的额外差异

SDS §5.2 与 §5.3 要求：

- 平台/接近平地目标 `0.075`；
- 楼梯目标 `max(0.17, h_step+0.02)`；
- 不能因为 tile 属于楼梯，就在平台上也用 0.22（SDS `:604`、`:651-658`）。

v07 的 `_clearance_targets`：

```python
# src/tasks/velocity/v07/rewards.py:115-123
heights = _STEP_HEIGHTS.to(tile_ids.device)...[tile_ids]
stair = maximum(0.17, heights + 0.02)
flat = tile_ids == FLAT_TILE_ID
return where(flat, 0.075, stair)
```

它按 **整个 tile** 选目标；楼梯 tile 的顶部平台仍用楼梯目标 0.22。r1 的 512 环境包含 `stairs_walk`/`legacy_l40`，因此该差异不是纯理论。

**判定：v07 与 SDS §5.3 的差异包括 3 个缺项、减速列两处、逐足抬脚目标、+2/+8/+10/−10/0 的一次性项、巡航门控和停车清零；r1 的优化目标与 S2 设计不一致。**

## 3. 影响范围与作废边界

### 3.1 S2.0 诊断不受影响（否认污染）

S2.0 诊断是推理链路，不加载优化器/训练状态：

- `scripts/run_a2_v09_s2_0.py:482` 明确注释 “Actor on the execution device; no optimizer/training state is loaded.”
- 执行循环 `scripts/run_a2_v09_s2_0.py:545` 只取 `_, done, info = env.step(raw_action)`，**丢弃 reward**；后续只使用 `done/info` 与事件指标。
- `src/tasks/velocity/v09/s2_execute.py` 与 `scripts/run_a2_v09_s2_0.py` 未消费 `s2_rewards`；S2.0 的报表/矩阵基于推理事件，不基于奖励。

因此，**S2.0 的诊断、矩阵、报表不因 r1 奖励接线错误而作废**；不需要重跑 S2.0 来“修正奖励”。但 S2.0 也不能用来证明 S2.1 的学习行为。

### 3.2 四个 S2.1-r1 检查点全部作废（确认）

文件、元数据与台账一致：

| 文件 | `learning_samples` | `ledger_cursor.rollouts` | SHA256（复核值） | 判定 |
|---|---:|---:|---|---|
| `s2_1_000005013504.pt` | 5,013,504 | 204 | `59cf153d510d605c8f2c718baeb9cb1864dd7d4da89444bfa846c5261d715665` | **作废** |
| `s2_1_000010027008.pt` | 10,027,008 | 408 | `429384deeded58d51dce20a7031a9a2b3bd39c8dd7387237ca5e8a0997640315` | **作废** |
| `s2_1_000015040512.pt` | 15,040,512 | 612 | `247cc0c47173120c2fea9cffb5cff87b821ea69685416d920d43410005aa703f` | **作废** |
| `s2_1_000020054016.pt` | 20,054,016 | 816 | `16a06ea3f3799bdc3784b68d19ca4ff85edc2ffbe183fbfa336c3c43b4ee45af` | **作废** |

四个文件的共同元数据：

- `schema=11`，`stage='S2'`，`course='S2.1'`；
- `sds_sha256=09fa2d4c…`（当前 SDS 文本 SHA）；
- `source_sha256=efa9caa8…`（`s1_least.pt` 入口）；
- `evaluation_samples=0`，`gates_passed=[]`，`gate_uses=0`，`final_uses=0`；
- `candidate_sha256` 分别为 `3a0ebe0c…`、`383debad…`、`1216218d…`、`11aacb33…`；
- `code_sha256=41d56278…`。

这些元数据证明“文件/课程/入口/样本数”，**不能证明奖励配方**：`code_sha256` 只计算 `src/tasks/velocity/v09/s2_train.py`（`s2_train.py:233-236`），不包含 `s2_train_real.py`、`s2_rewards.py` 或奖励参数。因此不能因为 `sds_sha256` 是当前 SDS 就误判奖励合规。

四个检查点均产生于 `1ec58f8` 之前（最后文件 mtime 23:40，修复提交 23:46:49），且 r1 训练路径已确认走 v07 `RewardComputer`，故：

**判定：四个检查点全部作废，不得用于门禁、终验、选模、行为结论或任何学习曲线结论；文件保留仅作事故证据。**

### 3.3 门禁代码未强制拒绝作废 SHA（新阻断）

作废记录 `docs/v09_s2_1_r1_invalidated.md:17` 写“`run_a2_v09_s2_gate.py` 不得接受这些 SHA”。但代码实际只检查课程：

```python
# scripts/run_a2_v09_s2_gate.py:36-44
def _checkpoint_identity(path, *, course):
    payload = s2_checkpoint.load_checkpoint(path, device='cpu')
    metadata = payload['metadata']
    if metadata.get('course') != course:
        raise ValueError(...)
    return dict(path=str(path), sha256=_sha(path), ...)
```

没有 SHA 黑名单、候选 allowlist、`reward_recipe_sha256` 或 `train_real_sha256` 校验。四个 r1 检查点的 `course` 都是 `S2.1`，所以如果直接传参，门禁会接受其身份。

当前未发生实际门禁消耗：

- `logs/a2_v09/s2/gate_dry_run/checkpoint-503504/S2.1_gate_plan.json:8` 引用了 `59cf153d…`，但同文件 `:40-41` 为 `physics_started=false`、`ledger_written=false`；
- 该 dry-run 的 `gate_admissions_up_to_40` spent/reserved 均为 0（同文件 plan 的 ledger_obligations）；
- 因此没有已发生的门禁结果或样本消耗被污染。

**判定：当前无实际门禁污染；但“作废仅写在文档、代码不拒绝”是重跑/门禁前的 P0 准入缺口，必须补代码层拒绝。**

### 3.4 检查点可追溯性缺口

`s2_train.py:233-236` 的 `code_sha256` 只哈希 `s2_train.py`；`s2_train_real.py` 的奖励适配器与 `s2_rewards.py` 不在身份内。建议后续检查点至少加入：

- `reward_recipe='S2_S5_v1'` 及 `s2_rewards.py` 文件 SHA；
- `s2_train_real.py` 文件 SHA；
- `reward_term_count=20` 与首次 rollout 的 `reward_terms` 名称；
- 训练开始时的 `git HEAD`/dirty 状态。

这不是本次作废结论的必要条件，但能避免同类事故再次“检查点看起来完全合规”。

## 4. 台账：失败消耗保留（确认）

### 4.1 账本链完整性

只读重算 `logs/a2_v09/s2/ledger.jsonl`：

- 事件数 **2,276**（含 header）；
- `sequence` 连续；
- `previous_sha256` 链连续；
- 每条 `event_sha256` 与 canonical JSON 的 SHA256 一致；
- 最后事件 SHA `fcac06958ef5bdc83d029fc4b54928169030bf611031224b5cebf258eeaa5a3c`。

**账本链通过。**

### 4.2 `learning_S2_1` 计数

| 事件类型 | 次数 | 样本数 |
|---|---:|---:|
| `obligation_reserve` | 826 | 20,299,776 |
| `obligation_commit` | 825 | **20,275,200** |
| `obligation_release` | 1 | 24,576 |

- 825 次 commit 对应 rollout 0..824（最后一次 `rollout824:commit`，`sequence=2273`）。
- 最后一次 reserve 是 `rollout825:reserve`（`sequence=2274`），随后 `orphan-release:stop` 释放 24,576（`sequence=2275`），原因明确写“invalid S2.1 run stopped (v07 reward path used instead of SDS section 5)”。
- 因此当前 **spent=20,275,200，reserved=0**；上限 800,000,000；余额 779,724,800，按 24,576/rollout 可跑 **31,727** 个完整 rollout（31,727×24,576=779,722,752，余 2,048 样本不足以再开一个完整 rollout）。

`logs/a2_v09/s2/s2_1.log:922` 的 816 rollout 与第四检查点一致，`:931` 的 825 rollout 与停训点一致；PID 1931607 当前不在运行，未发现新的 `train_a2_v09_s2` 进程。

**判定：确认该 20,275,200 为实际已提交的学习消耗；in-flight 24,576 仅释放预留，不退还已消耗样本；“失败消耗保留、reserved=0”成立。**

## 5. 对当前修复提交 `1ec58f8` 的独立复核（不批准重跑）

### 5.1 接线本身已改到 S2 批量实现（确认）

当前 `src/tasks/velocity/v09/s2_train_real.py`：

- `:187-189` 用闭包覆盖 `self.env._reward = _s2_reward`；
- `:353` 调用 `s2_rewards.compute_reward_batch(frame)`；
- 未再调用 `reward_computer`；`S2TrainEnv` 的 `step` 仍委托 `V07Env.step`，但 `_reward` 已被替换。

因此，**“训练仍使用 v07 RewardComputer”这一具体缺陷在 `1ec58f8` 的接线上已修复**。但训练路径实际调用的是 `compute_reward_batch`，不是标量 `compute_reward`；两者在 600 个随机帧的 CPU 对拍中一致（我另外做了 200×3 随机帧核对），16 条新增 CPU 测试也通过。**“标量函数被训练直接调用”仍应否认；“训练使用等价的 S2 批量函数”可以确认。**

### 5.2 仍然阻断的 P0 问题

以下均为 `1ec58f8` 的静态事实；未运行 GPU/物理，故不声称训练曲线表现，只判断语义合规性。

#### P0-1 适配器状态没有随 episode/rollout 重置

`S2TrainEnv.reset` 只重置底层环境：

```python
# s2_train_real.py:367-369
def reset(self):
    obs, _info = self.env.reset()
    return obs
```

但 `S2Trainer.collect_rollout` **每个 rollout 都调用 `self.env.reset()`**（`s2_train.py:129-130`）。适配器在 `:222-240` 创建的 `_s2_top_latched`、`_s2_success_latched`、`_s2_stop_requested`、`_s2_stop_step`、`_s2_hold_step`、`_s2_g_step`、`_s2_air_time`、`_s2_prev_contact`、`_s2_contacted`、`_s2_prev_qdot`、`_s2_steps` 全部保留。

后果：

- 第一个 rollout 之后 `top_latched/success_latched` 可能永久为真，后续 episode 不再支付 `+2/+8`；
- `stop_requested` 在后续 episode 一开始就为真，`flat_stop` 环境可能从第一步就进入 HOLD；
- `g_step`/`hold_step` 指向旧 episode 的步号，`g_level` 和 20 秒停车判定跨 episode 错位；
- `air_time`、`prev_contact`、`prev_qdot` 跨 episode 继承，腾空奖励与 `dof_acc` 被污染；
- `top_latched`/`success_latched` 没有随 episode 清零；v07 的 `state.success` 虽已由事件层保证 `success & ~failure`（`v07/events.py:330`），但 latch 仍会跨 episode 保留。

**P0。**

#### P0-2 `g_level` 对平地恒为 0，且用接触而非四足投影

适配器：

```python
# s2_train_real.py:254-263
support = env.climb.current_support.bool()
g_ready = support.all(-1) & (local[:, 0] > self._s2_front + 0.4) & ~failure
...
g_level = where(t0 < 0, 0, (steps - t0)/25)
```

但 v07 的 `current_support` 只在楼梯环境有效：

```python
# v07/events.py:311
stairs = tile_ids > 0
# v07/events.py:234
valid = candidate & feet_safe & stairs[:, None] & (surface_ids >= 0)
# v07/events.py:239
current_support = valid & (support_count >= SINGLE_FOOT_FRAMES)
```

平地角色 `flat_walk/flat_stop` 的 `tile_id=0`（`s2_train_real.py:23-30`），因此 `current_support` 恒为 false，`g_level` 恒为 0。SDS §5.2 `:595-600` 明确“普通平地正常活动区域：1；楼梯区：0”。这意味着平地上的 `pitch` 与 `horizontal_overspeed` 两项被实际关闭，且平地不会得到应有的水平面门控。

楼梯/平台侧还有两个问题：

- `support.all(-1)` 是**四足同时接触/去抖状态**，不是“四足投影均在同一个顶部平台内”；正常行走时摆动腿不接触，`all` 很难成立，导致 `g_level` 也很可能长期为 0；
- 没有校验四足 `surface_ids` 是否属于同一平台；`current_support` 每只脚独立，`.all()` 不能替代同平台判断。

**P0。**

#### P0-3 `stable_stand` 不是 §5.3 的稳定站立

适配器只做：

```python
# s2_train_real.py:352
stable_stand=((phase > 0) & (c_exec <= 1e-7))
```

SDS `:667-674` 要求同时满足水平速度、roll/pitch、总角速度、机身高度、四足真实支撑力与连续 2 帧、足点边界内缩 0.042m、四足平均水平速度等条件。当前实现只要“停车且指令为零”就付 `+1`，即使机器人还在晃动/摔倒边缘。**P0。**

#### P0-4 软关节越限项被硬编码为 0

```python
# s2_train_real.py:345
joint_limit_violation=torch.zeros(n, device=env.device).cpu().numpy()
```

SDS §5.3 第 18 项是 `-10 × 原上下限越界距离之和`；当前该项永远为 0，等于少了 20 项中的一项。**P0。**

#### P0-5 停车清空腾空累计未实现

批量函数返回 `clear_air_accumulator`（`s2_rewards.py:234`），但适配器从未读取它；`_s2_air_time` 只在接触时清零（`s2_train_real.py:303-307`）。SDS `:660-665` 要求停车请求发出时关闭腾空奖励并清空累计状态。当前 `feet_air_time` 只在打分时被置零（`s2_rewards.py:220`），累计器本身保留，恢复巡航后会带入旧腾空时间。**P0。**

#### P0-6 抬脚目标仍不是逐足真实表面

```python
# s2_train_real.py:309-315
on_platform = feet_local[:, :, 0] >= self._s2_front[:, None]
targets = where(on_platform, 0.075,
                where(self._s2_stairs[:, None],
                      max(0.17, self._s2_h[:, None] + 0.02),
                      0.075))
```

问题：

- `_s2_stairs` 来自 `task` 字符串包含 `stairs`（`:212-214`）；`legacy_l40` 的 task 是 `legacy_l40`（`s2_train_env.py:75-77`），**不含 `stairs`**，于是 51 个 L40 环境的全部足目标变成 0.075，而不是楼梯 0.22；
- `h` 是行级几何高度（`self._s2_h`），不是 `H(foot_xy)` 的逐足真实表面高度；
- 楼梯行中，位于楼梯前平地的足也会被算成楼梯目标 0.22，而不是 0.075；
- 没有处理 `surface_valid`/无效点合同。

SDS `:604`、`:651-658` 明确要求逐足按投影所在真实表面选目标，不能用行级/tile 级替代。**P0。**

#### P1 其他精度问题

- `dof_acc` 的 `joint_velocity_prev` 来自 `_s2_prev_qdot`，初始为 0（`:231`），第一帧会产生虚假大加速度；episode 重置后也不清零（与 P0-1 同源）。**P1。**
- `v_heading` 使用世界系 x 速度 `root_link_lin_vel_w[:,0]`（`:346`），而 SDS `:611` 要求“世界速度在当前水平朝向上的投影”；有偏航时不等价。**P1。**
- 每个控制步把所有奖励输入从 GPU `.cpu().numpy()` 再转回 GPU（`:322-355`），产生逐帧同步和主机传输；这是性能风险，不是语义结论，但在 512 env 训练中需实测。**P1（性能）。**

### 5.3 新增测试覆盖不足

`tests/test_v09_s2_rewards.py` 只测：

- 标量总公式与一次性项；
- `compute_reward_batch` 与标量在**一个基础帧**上一致；
- `g_level_at` 辅助函数。

它**没有**导入 `S2TrainEnv`，也没有覆盖：

- 连续两个 episode/rollout 后 latch 是否重置；
- 平地 `g_level=1`、楼梯 `g_level=0`、t0+1/t0+24/t0+25；
- `legacy_l40` 的抬脚目标 0.22；
- 真正的 stable_stand 条件；
- 非零 joint limit；
- 停车清空腾空累计；
- 失败帧的 `-10`、`+8`、`+2` 在真实适配器上的组合。

我运行了 `tests/test_v09_s2_rewards.py` 与 `tests/test_v09_s2_0_rewards.py`：**16 passed**（纯 CPU）。这证明公式函数本身与其单元测试一致，**不证明**训练适配器合规。

**判定：`1ec58f8` 修复了“接错到 v07”的接线，但没有满足 §5 的语义与状态生命周期；不批准 s2.1-r2 重跑。**

## 6. 结论与重跑准入条件

### 6.1 本次结论

1. **确认** r1 的奖励来自 `V07Env._reward → v07 RewardComputer`，不是 SDS §5；四个 r1 检查点全部作废。
2. **否认** S2.0 诊断被污染；S2.0 不需要因奖励问题重跑。
3. **确认** 台账 `spent=20,275,200 / 800,000,000`、`reserved=0` 是有效的失败消耗保留；余额 31,727 rollouts。
4. **否认** 门禁代码已经拒绝作废 SHA；这是文档与实现的缺口，重跑前必须补代码。
5. **否认** 当前 `1ec58f8` 已满足重跑准入；接线已改，但 P0-1…P0-6 仍未解决。

### 6.2 重跑前必须完成（独立复核口径）

- **状态生命周期**：在 `S2TrainEnv.reset()` 或每个 episode 的 `done` 掩码上重置适配器的 top/success/stop/g_level/air/qdot/prev_contact 状态；必须覆盖“每个 rollout reset”和“episode 中途 auto-reset”两条路径。
- **g_level**：平地恒 1，楼梯 0；顶部平台用四足 XY 投影同平台 + 机身超过前缘 0.4m 触发，0.5s（25 步）线性上升；失效归零；与 SDS `:595-600`、检查清单 `:22` 一致。
- **stable_stand**：实现 `:667-674` 的全部条件，而不是 `phase>0 & c_exec≈0`。
- **joint_limits**：接入真实软限位越界距离；不得恒 0。
- **air-time**：停车请求时清零累计状态，恢复后不得带入旧腾空时间。
- **抬脚目标**：逐足 `H(foot_xy)` 真实表面查询；平台/平地 0.075、楼梯 0.22；`legacy_l40` 必须按楼梯处理。
- **测试**：至少覆盖连续两个 episode 的 latch 重置、平地/楼梯/平台 `g_level`、L40 抬脚、真实 stable_stand、joint limit、停车清空、失败优先和 20 项分项手算；并保留“同一状态 S2 vs v07 分项对照”。
- **门禁**：`run_a2_v09_s2_gate.py` 增加作废 SHA 拒绝或候选 allowlist，并校验奖励配方身份；否则 r1 检查点仍可能被当作合法输入。
- **台账**：在独立复核通过前不再产生 `learning_S2_1` 新 reserve；重跑从 `s1_least.pt` 开始，attempt `s2.1-r2`。

## 7. 复核限制

- 未运行 GPU、mjlab/warp 物理、训练或门禁 execute；所有结论来自源代码、提交时间、文件 SHA、ledger 与 CPU 单元测试/随机对拍。
- `1ec58f8` 的适配器 `S2TrainEnv` 需要真实 `V07Env` 才能构造；本次未构造真实环境，因此 P0-1…P0-6 是静态代码路径判断，不是运行日志结论。
- 本次没有修改任何实现/测试/SDS/证据/ledger/checkpoint；只新增本报告与配套 JSON。
