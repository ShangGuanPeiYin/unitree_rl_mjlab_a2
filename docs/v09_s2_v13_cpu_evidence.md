# V13 S2.1 CPU 证据包（plan v13 §6.1 items 1–6）

- 生成角色：实现方 CPU 证据（不含 GPU / mjlab / MuJoCo / 训练 / gate）。
- 依据：`docs/v09_s2_1_numerical_acceptance_plan_v13.md` §2、§5、§6.1、§7。
- 独立判据：`logs/a2_v09/s2_0/verification/numerical_acceptance_handcalc_v13.json`（SHA256 `2d1a7679fc180989e6ff149c83fb730f1b7c6cacdc588b4783fc7c3cf5547943`，与 plan 声明一致；脚本会重算并断言）。
- 证据 JSON：`logs/a2_v09/s2/v13/v13_cpu_evidence.json`，SHA256 `72a32c7aed58949fd38037f0ddfc7b31b8716a91f593d2694d53ed03d8a352a8`。
- 代码快照：`HEAD=a362b1b661338b6fd1fe699fa72570b093efb42a`（`main`，含 lead 并行落地的 `2e7ed6e`/`2d31eeb`/`a362b1b` 三个修复）。为保证可审计，证据 JSON 的 `repo.production_file_sha256` 记录了本次实际读取的 11 个生产文件 SHA256。**若这些文件再次变化，必须重跑本脚本，不能沿用本报告的数字。** 本次 `s2_train.py` 快照 SHA256 为 `a3388aac861df1a220b10b8662e1071b613a384f466e1b529999e96d7b4516ed`；证据脚本自身 SHA256 记录在 JSON 的 `inputs.verify_script`。

## 1. 执行的命令

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2

# 证据包（全部测量均来自本次真实 CPU 运行）
/home/xxs/anaconda3/envs/a2/bin/python scripts/verify_a2_v09_s2_v13.py \
  --output logs/a2_v09/s2/v13/v13_cpu_evidence.json \
  --handcalc logs/a2_v09/s2_0/verification/numerical_acceptance_handcalc_v13.json

# 回归测试
/home/xxs/anaconda3/envs/a2/bin/python -m pytest \
  tests/test_v09_s2_v13_evidence.py tests/test_v09_s2_1_train.py tests/test_v09_s2_rewards.py -q
# 结果：48 passed（本快照 HEAD=a362b1b 下的最终一次运行）
```

脚本退出码为 0 表示 `all_passed=true`；任一证据检查失败时退出码为 1，但 JSON/报告仍会写出（诚实 FAIL，不隐藏）。

## 2. 逐项结果

| # | 检查 | 结果 | 关键测量 |
|---|---|---|---|
| 1 | `handcalc_13_cases` | PASS | 13/13 用例；每例 20 项 raw terms + continuous/forward/total + column/cruise/top/success/failure；float64 容差 `rtol=1e-12, atol=1e-12`；`max |Δt total| = 2.693e-13` |
| 2 | `g_level_sequence` | PASS | step 9/10/11/34/35/41 → `0, 0, 0.04, 0.96, 1, 0`；`s2_rewards.g_level_at` 与 `S2Lifecycle` 两条 CPU 生产路径均复现 |
| 3 | `ppo_storage` | PASS | 4 env × 48 step = 192 个唯一 `(env_id,step)` 样本；obs/actions/old_logp/adv/ret 逐样本对齐；Adam 更新 20 = `EPOCHS(5) × MINIBATCHES(4)`；每样本被覆盖 5 次 |
| 4 | `gae_boundary` | PASS | 2 env × 4 step；“末步未终止→bootstrap V(next)”“末步终止→0”“中途 done 不跨 reset”三边界；独立 numpy 复算误差 0.0（≤1e-12）；`collect_rollout`→`gae` 静态契约 4 项全部命中 |
| 5 | `exception_settlement` | PASS | 真实临时 `S2Ledger`：reset 抛 0/96、step 抛 6/96（前 3 步真实执行）、update 抛 96/96、checkpoint 写抛 96/96；reserve/commit/release 与物理执行样本逐一相等 |
| 6 | `r1_rejection` | PASS | 四个 r1 文件真实 SHA 与登记一致且被拒；四个 dummy 注入 SHA 均被拒；CLI 默认 `--attempt-id s2.1-r2`；gate/resume/trainer 三处调用拒绝函数 |
| 7 | `flat_anchor` | PASS | JSON 三个手算例（`0.01`、`0.025`、`0`）被 `sample_numerator`/`AnchorAccumulator` 精确复现；当前 `S2Trainer.update` 为模式 (a)：微批累计分子、逻辑 minibatch 内统一除一次 |

`all_passed = true`（本次快照）。注意：§6.1 item 6 要求的是“锚定聚合合同”，本检查只验证 CPU 手算与聚合代码结构；真实 teacher/梯度属于 Run D，未验证。

## 3. 各项细节与证据位置

### 3.1 手算 13 用例（§6.1 item 1）

- 输入→生产 frame 的键映射固定在脚本 `HANDCALC_INPUT_MAP`（`v→velocity_b`、`omega→omega_b`、`g→gravity_b`、`q/q0/qdot→joint_*`、`a/a_prev→action/previous_action`、`qt→q_target_applied`、`contact→contact_filtered`、`fv/fh→foot_velocity_xy/foot_height`、`target→target_clearance`、`z_base/H→base_height/surface_height` 等）；JSON 中每个用例的输入键集合与映射表完全一致。
- 生产函数 `s2_rewards.compute_reward_batch` 全部以 numpy `float64` 运算（`np.asarray(..., dtype=float)`），因此采用 plan §2.2 的 float64 行，而不是 float32 行。
- `expected_s2` 只从 JSON 读取；脚本不调用任何生产奖励函数生成期望值。JSON 的 `expected_v07` 仅作为对照输出（`v07_difference_report`），并在 JSON 中明确 `v07 is NOT a judge`。
- 每一例的 20 项 raw term、`continuous`、`forward`、`total`、`column`、`cruise`、`top`、`success`、`failure` 全部记录 expected/produced/误差/passed。
- 本次测得：最大单项绝对误差 ≤ `1e-12`；`max |Δt total| = 2.693e-13`；最大单项相对误差是 `heading` 的 `5.17e-12`（绝对误差 `2.58e-14`，来自 JSON 期望值保留 12 位小数，仍在 `atol=1e-12` 内）。

### 3.2 g_level 序列（§6.1 item 1/§5.2）

- 主判据：生产纯函数 `s2_rewards.g_level_at(step, t0, ramp_steps=25)`。
- 交叉判据：生产 CPU 生命周期 `s2_lifecycle.S2Lifecycle` 逐帧驱动到 step 41（step 41 置 failure），同样得到 `0,0,0.04,0.96,1,0`。
- `pitch_term`/`overspeed_term` 按 JSON note 的文档公式由 `g_level` 推出（`gx=0.1`、`v_heading=0.8`、`c_exec=0.5`），并在 JSON 中标注 `source=documented_formula`。
- 观察（不计入本项 PASS/FAIL，供复核）：训练适配器 `s2_train_real.py:382` 的内联 `g_ready` 只含四足支撑、body margin、`~failure`，不含 `S2Lifecycle` (`s2_lifecycle.py:87`) 的 `top` 项；两者在“未到顶但四足支撑”时会给出不同 g_level 起点。该差异需要真实 rollout 才能判定影响，已写入 JSON 的 `adapter_contract_note`。

### 3.3 PPO 存储（§6.1 item 2）

- 用唯一标签构造 `transitions`：`label = 1000*env_id + step`，obs/actions/old_logp/adv/ret 各加一个百万级偏移以便同时验证“值”和“顺序”。
- 直接复用 `S2Trainer.update` 的 flatten 表达式与 `s2_train.flatten_time_major_old_logs`，并复现 `logical_minibatch_plan`/minibatch 循环。
- 结论：`flat_obs/flat_actions/flat_old/flat_adv/flat_ret` 在 `position = step*num_envs + env_id` 上逐样本一致；192 个样本标签唯一且全部对齐。
- 反例：env-major old-logp flatten 与 time-major 其它四元组在 192 个位置中有 190 个不一致，证明 time-major 不是可选项。
- Adam 次数：`EPOCHS*MINIBATCHES = 5*4 = 20`；每个样本每个 epoch 恰好被一个逻辑 minibatch 覆盖一次（覆盖计数 5）。
- 每个样本的 `position/env_id/step/label/五个数组值/minibatch_id` 全部写入 JSON。

### 3.4 GAE 边界（§6.1 item 3）

- 手算表：env0 `rewards=[0.1,0.2,0.3,0.4]`、`values=[0.5,0.6,0.7,0.8]`、全程未 done、`last_value=0.7`；env1 `rewards=[1,2,3,4]`、`values=[0.1,0.2,0.3,0.4]`、`terminated=[F,T,F,T]`、`last_value=0.0`（因末步 done）。
- 脚本内的独立 numpy 实现（不调用 `s2_runner.gae`）复算 advantages/returns，与生产 `s2_runner.gae` 的最大绝对误差为 `0.0`。
- 边界扰动探针（生产函数）：`last_value+1` 使 env0 末步 advantage 增加 `0.99 = GAMMA`；把 env1 `v[2]+10` 对 t=1（done 行）的 advantage 影响为 `0`；把 env1 `last_value+9` 对 advantage 影响为 `0`。
- 静态契约（本次行号，JSON 记录）：`s2_train.py:232` 传 `terminated`；`:233` 把 `truncated|failure|timeout` 合并为 `timeout`；`:234` `sampling_boundary` 全 True；`:235` `last_value=0 if done[-1] else V(next_obs)`。
- 合同说明：GAE 不区分 terminated/truncated/failure 的 bootstrap，三者都等价于 done→0；满足 plan §1.3 的“零 bootstrap”公式要求，但 Q035 要求的逐类分离仍只能由真实 rollout 的 per-env mask 证据回答（Run A）。

### 3.5 异常结算（§6.1 item 4）

使用真实 `S2Ledger` + `S2LearningBudget`（每轮预留 `2*48=96` 样本，仅写入 `tempfile` 目录；生产 `logs/a2_v09/s2/ledger.jsonl` 未触碰，测试中会比较其 SHA 前后不变）。以下行号为本快照：`s2_train.py:163`（`_rollout_steps=0`）、`:165`（`env.reset()`）、`:206`（`_rollout_steps += 1`）。

| 注入点 | 物理执行样本 | reserve | commit | release | learning_samples |
|---|---:|---:|---:|---:|---:|
| `env.reset` 抛 | 0 | 96 | 0 | 96 | 未更新 |
| `env.step` 第 4 次调用前抛（前 3 步已执行） | 6 | 96 | 6 | 90 | 未更新 |
| `update` 抛（rollout 已完成） | 96 | 96 | 96 | 0 | 未更新 |
| checkpoint 写入失败（rollout+update 已完成） | 96 | 96 | 96 | 0 | 96 |

- `collect_rollout` 在 `env.reset()` 之前把 `_rollout_steps=0`（`s2_train.py:163` vs `:165`），因此 reset 抛时 commit 为 0 而不是上一轮的 48。
- 已知缺口（JSON `known_gap`）：若 `env.step` 已经完成物理步进（`s2_train.py:176`）、异常发生在之后的 reward/append/转换中，`_rollout_steps` 尚未 `+1`（`s2_train.py:206`），台账会少计一帧。本证据注入的 step 场景是“第 k+1 次调用前抛”，所以台账与实际物理步一致；上述“执行后抛”的一帧误差未被本场景覆盖，也未修复。

### 3.6 R1 拒绝链（§6.1 item 5 / §7）

- 四个作废 SHA 从 `docs/v09_s2_1_r1_invalidated.md` 解析，并与手算 JSON `r1_registration`、`s2_r1.R1_INVALID_PREFIXES` 交叉核对。
- 四个真实文件（`logs/a2_v09/s2/s2_1/s2_1_*.pt`，只读）实测 SHA 与登记完全一致，`reject_invalid_checkpoint` 全部抛 `ValueError`。
- 另外创建 4 个 dummy 文件并注入四个作废 SHA 摘要，`reject_invalid_checkpoint` 全部拒绝（JSON 同时记录 dummy 文件的真实 SHA，明确这是摘要注入测试，不是伪造文件）。
- 训练 CLI 默认 `--attempt-id s2.1-r2`（`scripts/train_a2_v09_s2.py:48`），`reject_invalid_attempt('s2.1-r1')` 抛错，默认值通过；CLI 还显式拒绝 r1 输出目录。
- 静态调用点（本次行号）：gate `scripts/run_a2_v09_s2_gate.py:361`；CLI attempt `scripts/train_a2_v09_s2.py:53`；CLI resume `scripts/train_a2_v09_s2.py:142`；`S2Trainer.resume` `src/tasks/velocity/v09/s2_train.py:479`。
- 观察：`R1_INVALID_SHA256` 中三个键是 8 字符截断值，永远无法与 64 字符 digest 相等；真正的拒绝由 `R1_INVALID_PREFIXES` 完成，四个前缀覆盖完整。属于代码整洁性问题，不影响拒绝结果，本次未修改。

### 3.7 §5.4 平地锚定（§6.1 item 6）

- `s2_loss.sample_numerator` / `AnchorAccumulator` 对 JSON 三个手算例：
  - 单 clean（`mu_s=0.1`，差 0.1）：per-sample `0.01`，`n_clean=1`，均值 `0.01`，`loss=0.001`；
  - 两 clean（差 0.1/0.2）：per-sample `0.01/0.04`，均值 `0.025`，`loss=0.0025`；
  - 零 clean：均值 `0`，`loss=0`。
  均在 `1e-12` 内一致。JSON 第一例的 `clean` 有两个元素但 `mu` 只有一行；脚本把该行复制到非 clean 位置（被 mask 掉，不影响均值），并在 JSON 的 `reconstruction_note` 中说明，未改动 JSON。
- `S2Trainer.update` 聚合合同（运行时动态检测，本次快照为 **模式 (a)**）：
  - 逻辑 minibatch 循环 `src/tasks/velocity/v09/s2_train.py:348`；
  - `n_clean = int(flat_clean[index].sum())`（按当前逻辑 minibatch 统计）`:352`；
  - 微批内累加 `anchor_numerator` `:387-388`；
  - 微批结束后统一 `anchor_loss = beta * anchor_numerator / (12.0 * n_clean)` 并 `backward()` `:392`。
  符合 SDS:690-695。
- **重要时间线**：本次会话开始时工作区还是 plan §1.4 描述的模式 (b)（微批内用全局 rollout `n_clean` 除）；随后 lead 在并行修改中替换为模式 (a)。本证据 JSON 只对最终快照 (a) 做了真实测量；没有为 (b) 快照生成证据文件。脚本会动态读取源码：若回退到 (b)，`flat_anchor` 会诚实 FAIL 并在 JSON 写入 `defect`（不给 lead 修 bug，只报告）。

## 4. 仍被阻断的事项（需要真实后端）

以下 **未在本次执行**，仍是签字阻断项（plan §6.2）：

- **Run A**：真实 512/1024 后端两连续 sampling block（逐帧 `(env_id,step)`、reset mask、old/new logp、advantage/return、adam_step）。
- **Run B**：真实代表样本（flat cruise、DECEL/HOLD、stairs g_level t0 序列、platform 直启、L40 同帧 top+success、物理 failure、hard timeout）。
- **Run C**：真实 stop 序列动作/命令对齐（`a_request/a_applied/pd_target/c_request/c_exec` 同源）。
- **Run D**：`Frozen6400Teacher`（旧图 48×32）真实锚定、`n_clean>0`、`L_anchor` 与梯度。
- **Run E**：修复后执行栈的 512 资源/预算复核与 ledger 对账。

## 5. 明确未验证的内容

- 未运行 GPU、mjlab、MuJoCo、训练或 gate；所有证据都是 CPU 纯函数/临时台账/静态源码检查。
- 未验证真实 rollout 的 `(env_id,step)`、旧 logp、advantage/return、`reset` 跨 episode 状态清理。
- 未验证真实 terminated/truncated/failure_class 逐 env mask；GAE 目前只证明“统一零 bootstrap”的公式与静态调用契约。
- 未验证 §5.4 锚定在真实权重上的数值/梯度；当前 CLI 的 `teacher` 仍为 `None`（真实后端分支才会构造 `Frozen6400Teacher`）。
- 未验证 `a_request/a_applied/pd_target/c_request/c_exec` 的真实同帧数据；手算用例只覆盖 `compute_reward_batch`，不覆盖 `S2TrainEnv._s2_reward` 的完整 frame 组装（需要 mjlab）。
- 未验证 reset/checkpoint 与 ledger cursor 在真实断点恢复中的对账；仅验证了注入异常时的 reserve/commit/release 金额。

## 6. 复现与回归

- 证据脚本：`scripts/verify_a2_v09_s2_v13.py`（`--output`、`--handcalc`、`--tmpdir`）。
- 回归测试：`tests/test_v09_s2_v13_evidence.py`（手算加载/自洽、g_level、PPO 对齐、GAE 独立复算与扰动、锚定手算与聚合文档化行为、r1 拒绝、异常结算、生产 ledger 不被改写）。
- 生产文件一旦变化，先重跑证据脚本；`all_passed` 与各检查的 measured/expected/误差以 JSON 为准。
