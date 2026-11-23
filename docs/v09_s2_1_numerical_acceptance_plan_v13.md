# V13：S2.1 训练链路数值验收设计与独立手算参照

- 复核角色：A2-S2-VERIFY（独立于实现方）
- 复核方式：**纯 CPU、只读设计**；未运行 GPU/物理/训练，未修改实现、测试、SDS、证据、ledger、checkpoint。
- 依据：DSH-Q035 `doc/sds/reviews/v09_execution_rulings.md:336-342`；SDS v1.2 §5、§5.4、§6；`docs/v09_s2_1_r2_execution_contract.md`。
- 审计快照：`HEAD=6fe039e`（2026-11-22 23:55:52）；P0 已提交 `f20d8de`，当前 `scripts/train_a2_v09_s2.py` 将锚定 `teacher=None` 明确禁用。工作区仍有 `s2_physics.py` 等未提交改动，不在本计划审查范围。
- 配套手算：`logs/a2_v09/s2_0/verification/numerical_acceptance_handcalc_v13.json`（SHA256 `2d1a7679fc180989e6ff149c83fb730f1b7c6cacdc588b4783fc7c3cf5547943`；本文件生成后若再改 JSON 必须重算此 SHA）。
- 结论：**这是重跑前必须执行的验收设计，不是重跑批准**。当前 P0 尚未满足 Q035 §4；独立手算参照已给出，需实现方在真实后端产出证据后另行签字。

## 0. 当前状态与本次验收目标

P0 修复把“每轮 reset / old_logs 对齐 / GAE last_value / 异常结算”推进了一步，但**测试转绿不等于真实链路通过**。本计划把“可签字的数值验收”拆成五类证据：

| 类别 | 要回答的问题 | 是否可只看 CPU 单测 |
|---|---|---|
| A. 奖励公式 | §5 的 20 项、两列、0.02、位移、一次性项是否逐位正确 | 可以，用手算 JSON |
| B. PPO 存储 | `(env_id, step)`、old logp、GAE、终止/reset 边界、flatten、minibatch/Adam 是否正确 | 部分可以；真实 rollout 必须另采 |
| C. 平地锚定 | §5.4 是否真正用原始 6400/旧图/clean mask 反传 | 不能；必须真实 teacher/梯度 |
| D. 动作与命令 | `a_request`/`a_applied`/PD 目标/`c_exec` 是否对齐 | 不能；必须真实步进记录 |
| E. r1 拒绝链 | r1 四检查点及派生链是否在代码层被拒绝 | 可以，用拒绝测试 |

本计划要求**先 A/E，再 B/C/D**；A 不通过则无需跑 GPU；E 不通过则不得启动 r2。

## 1. 真实调用链审计（当前工作区）

### 1.1 训练入口与环境层次

- `scripts/train_a2_v09_s2.py:119-131` 构造 `S2TrainEnv` → `S2Trainer`；`teacher` 在非 fake 后端时传入（当前 CLI 已加 `V09ActorCritic.migrate_6400_checkpoint`，见 §1.4）。
- `S2TrainEnv` 是 `V07Env` 的**包装/组合**（`s2_train_real.py:168-193`），不是子类；`self.env._task_state` 和 `self.env._reward` 被替换。
- `s2_physics.py` 的 `S2PhysicsEnv` 是**执行/验收装配**（`s2_physics.py:740-838`），不是训练奖励路径；训练奖励只经过 `S2TrainEnv`/`V07Env`。因此 `s2_physics.BackendStep` 的 `a_request/a_applied/c_exec` 可作为动作对齐的参考观测，但不能把 `s2_physics` 当成 PPO 的训练环境。

### 1.2 `collect_rollout` 的 episode 连续性与 reset

当前代码已改为不每轮全量 reset：

```python
# src/tasks/velocity/v09/s2_train.py:135-143
if self._obs is None:
    self._obs = self.env.reset()
obs = self._obs
...
self._rollout_steps = 0
```

`S2Trainer` 每轮保留 `_obs`；每个 rollout 48 步（0.96 s）不再强制重置，满足 Q035 的“episode 跨 rollout 连续”。**这一层通过。**

但还有三个必须由真实 rollout 证明的点：

1. `S2TrainEnv` 的 S2 适配器状态并不能仅靠 `task_state` 全部清理。`_s2_task_state` 在 `episode_length_buf <= 1` 时清 `top_latched/success_latched/top_event`（`s2_train_real.py:227-234`），但 `_s2_stop_requested`、`_s2_hold_step`、`_s2_g_step`、`_s2_air_time`、`_s2_prev_contact`、`_s2_contacted`、`_s2_prev_qdot`、`_s2_steps` 仍未按 episode 清（见 `s2_train_real.py:274-292` 的 lazy init、`:335-358` 的使用）。这会让 stop、air-time、dof_acc 跨 episode 污染。
2. `S2TrainEnv.reset()` 仍只调用 `self.env.reset()`（`s2_train_real.py:420-421`），没有显式清理上述适配器状态；跨 rollout 的 episode 连续性因此只完成了“obs 连续”，没有完成“奖励状态连续的正确边界”。
3. `_s2_task_state` 只用 `episode_length_buf <= 1` 判定 episode 首帧；真实后端在 auto-reset 后是否保证 `episode_length_buf == 0/1`，必须用 `(env_id, reset_reason)` 日志验证，不能假设。

**验收要求**：两次连续 sampling block（96 步）中，至少一个环境跨越一次终止/auto-reset；逐帧记录 `episode_length_buf`、`done`、reset mask、上述全部 S2 状态，证明 reset 后旧锁存为 0、未 reset 环境不受影响。

### 1.3 GAE 边界 `V(next)` 与 terminated/truncated 语义

当前 `collect_rollout` 已加边界 bootstrap：

```python
# s2_train.py:174-188
next_values = self.model.evaluate(self._obs).squeeze(-1)
...
last_value=(0.0 if done[index,-1] else float(next_values[index]))
```

`s2_runner.gae` 在最后一步 `boundary & ~done` 时使用 `last_value`；内部 done 行 `non_terminal=0`，不跨 reset（`s2_runner.py:105-127`）。**边界公式本身正确**，但还需验证：

- 最后一步 `done=True` 时，`last_value` 必须为 0；最后一步 `done=False` 时必须等于 `V(next_obs)`。
- 中间 done 行后不得跨 reset 使用 `v[t+1]`；当前 `non_terminal=0` 满足，但 `transitions` 的 `done` 是否逐环境正确必须由 `(env_id,step)` 证明。
- 当前 `collect_rollout` 仍把 `terminated=np.zeros(...)`、`timeout=done` 传给 `gae`（`s2_train.py:184-186`）；虽然两者在 `gae` 内部都作为 `done` 零 bootstrap，但 Q035 要求 `terminated/truncated/failure_class` 分离，**不能只靠“公式上等价”**。`info` 中虽新增 `terminated_mask/truncated_mask/failure_mask/timeout_mask`（`s2_train_real.py:423-440`），但 `S2Trainer` 尚未在 GAE 中使用这些 mask；需要明确“value target 统一零 bootstrap”的合同，并在验收证据中逐行列出类别，而不是把全部 done 都标为 timeout。
- `S2TrainEnv.step` 的 `info['failure_class']` 是**整个 batch 的单字符串**（`:439`），不是逐环境类别；Q035 要求逐例失败类别，训练证据必须另存 per-env mask/class。

**通过条件**：固定 `(env_id, step)` 的 reward/value/done 表，独立重算 GAE 与 return；至少覆盖“最后一步未终止”“最后一步终止”“中途 done 后下一 episode”三种边界，误差见 §2.2。

### 1.4 §5.4 平地锚定项（当前仍不满足）

代码已有锚定片段：

```python
# s2_train.py:200-204
role_rows = (getattr(self.env, 'role_plan', {}) or {}).get('envs') or []
clean_env = ... task.startswith('flat_') ...
# s2_train.py:253-262
if self.teacher is not None and n_clean:
    mu_student = self.model(clean_item)
    mu_teacher = self.teacher(clean_item).detach()
    anchor = sum((mu_student-mu_teacher)**2) / (12.0 * n_clean)
    scaled_loss = scaled_loss + self.beta_flat * anchor
```

但存在四个阻断：

1. **真实 `S2TrainEnv.role_plan` 没有 `envs` 键**：它在 `s2_train_real.py:158-160` 只保存 `counts/order`；`role_rows=[]`，`clean_env` 全为 False，`n_clean=0`，真实训练锚定项恒为 0。P0 测试用的 `S2FakeTrainEnv` 的 `role_plan` 有 `envs`，所以测试绿但真实链不绿。
2. **clean 判定不是 §5.4 的 clean-flat**：`task.startswith('flat_')` 只是角色前缀，包含 stop 等；SDS 要求普通平地 × 名义动力学 × 无噪声/延迟/外力（`s2_loss.clean_flat_mask` 的五个 flag，`s2_loss.py:10-18`），当前环境未提供这些 flag。
3. **教师来源/接线错误**：`6fe039e` 已把 CLI 的 `teacher` 设为 `None`（`scripts/train_a2_v09_s2.py:123-126`），注释明确说“40×30 迁移模型不得冒充 48×32 原始 6400，锚定暂时禁用”。因此当前真实链**根本没有锚定**。SDS §5.4 要求“教师始终为原始 6400，使用旧图”（SDS:697）；原始 48×32 是 `Frozen6400Teacher`（`model.py:182-203`），训练观测是 40×30，必须另取旧图 teacher observation，不能用迁移后的 40×30 模型冒充。
4. **锚定聚合错误**：SDS:690-695 要求各微批只累计**分子**，最后统一除以整个逻辑 minibatch 的 `n_clean`；当前代码在每个 micro 内用**全局 n_clean** 除一次，再乘 `len(micro)/len(index)` 进入 backward，既不是全局分子聚合，也不是正确的梯度权重。`mu_student = self.model(clean_item)` 还未确认返回 12 维 action mean。P0 测试只断言 `teacher.calls > 0`（`tests/test_v09_s2_1_train.py:446-457`），没有校验数值/梯度/clean mask，不能作为通过证据。

**验收要求**：真实链上至少一份 `teacher=Frozen6400Teacher`（旧图 48×32）、一份 clean flag 表、`n_clean>0`；用独立手算给出 `L_anchor`（§2.3）并对比反传梯度；`clean` 不得包含楼顶/扰动病例。当前状态：**FAIL**。

### 1.5 time-major / env-major flatten

当前已把 old logp 改为 time-major：

```python
# s2_train.py:45-46
def flatten_time_major_old_logs(transitions):
    return torch.stack([row['log_probs'] for row in transitions], 0).reshape(-1)
```

`flat_obs` 按 transitions（step）cat，`flat_actions` 按 step stack，`advantages/returns` 由 `[N,T]` 转 `[T,N]` 后 flatten；旧 logp 现在同序。**这一处通过。**

但现有测试只比较 helper 的列表（`tests/test_v09_s2_1_train.py:434-443`），没有在完整 `update` 中证明 `(env_id, step)` 对齐。必须用带唯一标签的 fake 样本做 `flat_obs`/`flat_actions`/`flat_old`/`flat_adv`/`flat_ret` 五元组核对，且至少一个 env 在同一 step 的 logp 唯一。

### 1.6 `train_rollout` 异常结算

当前代码：

```python
# s2_train.py:273-289
budget.begin(...)
try:
    ...
except Exception:
    actual = num_envs * self._rollout_steps
    budget.finish(attempt_id, actual)
    raise
```

风险：

- `_rollout_steps` 在 `collect_rollout` 内、`env.reset()` **之后**才置 0（`s2_train.py:138-143`）；若第 N 轮 `env.reset()` 抛异常，`_rollout_steps` 仍是上一轮的 48，`actual` 会被算成整轮，而不是本轮实际 0。
- 若 `env.step` 已执行但 append/转换抛异常，`_rollout_steps` 尚未 +1（`:169`），会少计一帧；Q035 要求按实际物理执行收口。
- 现有 `test_learning_budget_releases_on_rollout_exception`（`tests/test_v09_s2_1_train.py:130-159`）用 `CrashingTrainer` 覆盖“collect 一开始就抛”的 0 消耗，未覆盖 reset 抛、部分 step 抛、update 抛三种情况。
- 异常时 `budget.finish` 已 commit 实际样本，但 `self.learning_samples` 不增加（`:268-269` 仅在 update 成功后）；恢复时的 checkpoint cursor 需要与 ledger 实际 commit 对账，不能静默不一致。

**验收要求**：四种异常注入（reset/step/update/checkpoint）分别预期 reserve/commit/release 事件与 `actual` 精确匹配；`learning_samples` 与 ledger 在异常后要么一致，要么明确记录“未入 PPO”的差异。

### 1.7 动作与命令链

- 训练 frame 的 `action` 是 `action_manager.action`（策略请求），`q_target_applied`/`a_applied` 是 `action_term._processed_actions`（实际 PD 目标）（`v07/env.py:737-744`、`692-693`）。
- `S2TrainEnv._s2_reward` 的 `c_request`/`c_exec` 目前仍从 `env.sample_command` 读（`s2_train_real.py:319` 附近），并在 reward 末尾把 `env.command[:,0]=c_exec`（`:411`）。由于 `V07Env._task_state` 会在下一步把 `sample_command=command.clone()`，`sample_command` 已经被上一帧改写为 `c_exec`；在 DECEL 中会出现**请求值随执行值下降、ramp 二次衰减**，`c_request` 也不再是原始请求。S2.1 无 stop 时可能不暴露，S2.2/S2.3 stop 必暴露。
- `s2_physics.py` 的 `BackendStep` 提供 `a_request/a_applied/pd_target/c_exec/c_request`（`:817-821`），但它是执行包装，不是训练环境；动作对齐必须从训练环境同帧采集，不能拿 `s2_physics` 的字段替代训练 frame。

**验收要求**：真实 stop 序列中 `c_request` 保持计划速度，`c_exec` 单调 ramp 到 0；`a_applied==pd_target`；`a_request` 只作为策略请求，不冒充 PD 目标；奖励 frame 的 `c_exec` 与 actor 下一步实际消费的 command 同源。

## 2. 独立手算参照

### 2.1 独立性声明

`numerical_acceptance_handcalc_v13.json` 的期望值由**独立 stdlib 实现**按 SDS §5.1/§5.3 生成，**没有导入/调用** `s2_rewards.py`、`s2_train_real.py`、`v07/rewards.py` 或任何生产奖励函数。手算输入状态、权重、公式和期望值均固化在 JSON；验收时应从 JSON 读取输入，另写一份独立计算（或人工手算）核对，不得用生产函数当 expected。

### 2.2 浮点容差（先声明）

| 比较 | 容差 |
|---|---|
| 手算 JSON 的独立复算（float64） | `rtol=1e-12`, `atol=1e-12` |
| 生产 float32 单 term vs 手算 | `rtol=2e-6`, `atol=2e-7` |
| 生产 total vs 手算 total | `atol=2e-6`（必要时 `rtol=2e-6`） |
| 离散量 | `column`、`cruise`、`decel`、`top`、`success`、`failure`、`timeout` 必须**精确相等** |
| 事件金额 | `+2/+8/-10` 精确到 `1e-12`（float64 输入） |

若生产实现使用 float64，单 term 也应按 `1e-12` 检查；若使用 float32，以上容差是上限，不能用“float32 误差”掩盖权重/项数错误。

### 2.3 覆盖用例

JSON 共 13 个用例（12 个标量状态 + 1 个无事件复制），每个用例给出完整输入、20 项 raw terms、continuous、forward、top/success/failure 金额和 total；同时给出 v07 期望用于同帧差异对照。

| 用例 | 目的 | 关键断言 |
|---|---|---|
| `flat_cruise_full20` | 平地巡航、20 项全量、cruise 列 | 20 项均列出；`continuous=0.02Σw f`；`forward=0.4*dx` |
| `decel_partial_command` | DECEL 且 `c_exec>0` | 前进=0；air=0；`ang_vel_xy` 用 -0.20；stable=0 |
| `hold_stable_zero` | HOLD 且 `c_exec=0`、stable 标志 | 前进=0；air=0；stable=+1；`ang_vel_xy` -0.20 |
| `climb_g_zero` / `climb_g_mid` / `climb_g_one` | g_level 0/0.4/1 | pitch=`-5·g·gx²`；overspeed=`-2·g·max(...)²` |
| `platform_direct_success` | 平台直启 | 无 top +2；success 只 +8 |
| `l40_top_success_same_frame` | L40 同帧 top+success | +2 与 +8 都付，失败不抑制 |
| `failure_suppresses_positive` | 物理失败同帧带 top/success | 只 -10；不付 +2/+8 |
| `timeout_pays_failure` | 超时 | 按失败 -10；不付 +2/+8 |
| `top_only` / `success_only` | 同帧单个正事件 | S2 top 只 +2；success 只 +8；v07 差异见 §2.5 |
| `no_positive_event` | 无正事件 | 无一次性项；只连续+前进 |
| `decel_partial_command` 复用 | dt 一次 | 连续项不逐项乘 dt；forward 不再乘 dt |

另外 JSON 的 `g_level_sequence` 固定 `t0=10` 的 `step=9/10/11/34/35/41`，期望 `0,0,0.04,0.96,1,0`；`flat_anchor_examples` 固定 clean/非 clean 的 `L_anchor` 手算（`0.001`、`0.0025`、`0`）。

### 2.4 计算骨架（必须按这个顺序）

```text
cruise = c_request > 0 and phase not in (DECEL, HOLD) and |c_exec-c_request|<=1e-9
column = 1 if phase in (DECEL, HOLD) else 0
continuous = 0.02 * sum_i w_i[column] * f_i
dx_used = 0 if decel else dx
forward = 0.4 * dx_used * cruise
failure = failure or timeout
top = top_once and not failure
success = success_once and not failure
total = continuous + forward + 2*top + 8*success - 10*failure
```

注意：

- `feet_air_time` 在 DECEL/HOLD 的 raw term 先置 0；`stable_stand` 只在 `decel and |c_exec|<=1e-7 and stable_stand` 时为 1；`foot_clearance` 的目标由**逐足真实表面**给出（手算 JSON 用显式 target 列表，生产实现必须来自 `H(foot_xy)`，不是行级 h/任务字符串）。
- `0.02` 只乘一次；`0.4*dx` 不乘 `0.02`。
- 同帧失败优先；超时等价失败；失败帧不得同时领取 top/success。

### 2.5 v07 同帧差异（证明旧链失效）

手算 JSON 对 `flat_cruise_full20`、`decel_partial_command`、`timeout_pays_failure`、`success_only`、`top_only` 给出 v07 期望：

- v07 17 项、`ang_vel_xy=-0.05`、`feet_air_time=+0.2` 且不因 stop 清零；
- v07 成功 +10、失败 -10、超时 0、无 top +2；
- v07 前进 `0.4*dx` 每步都付；
- v07 无 pitch/overspeed/stable 三项。

验收时必须报告“同一状态 v07 vs S2”的 term/total 差，且以本 JSON 的 S2 手算为判据，不以生产 v07 函数为判据。

## 3. 覆盖矩阵（公式/真实路径/证据）

| 覆盖项 | 手算用例 | 真实路径代表样本 | 必须保存的证据 | 通过判据 |
|---|---|---|---|---|
| 平地巡航 | `flat_cruise_full20` | `flat_walk`，`vx=0.3/0.5/0.6` 各一帧 | `(env_id,step)`, 20 raw, total, `c_exec` | 20 项 + total 在容差内 |
| DECEL | `decel_partial_command` | `flat_stop` ramp_2s 的第 1 个 DECEL 帧 | 两连续帧的 `c_exec`、phase、air、stable | 前进 0、air 0、ang -0.20、stable 0 |
| HOLD | `hold_stable_zero` | 同一 stop 的 c_exec=0 首帧 | stable 五条件原始量、+1 | stable=1 仅在条件全真 |
| 登阶/g_level | `climb_g_zero/mid/one` + `g_level_sequence` | `stairs_walk` t0、t0+1、t0+24、t0+25、失效 | g_level 逐帧、pitch/overspeed raw | 0→0.04→0.96→1→0；同帧 g |
| 平台直启 | `platform_direct_success` | `platform_walk` spawn 后一帧、post 距离后一帧 | top/success flags | 无 top +2；success +8 |
| L40 同帧 | `l40_top_success_same_frame` | `legacy_l40` 顶平台同帧 | top/success/failure 三 mask | +2/+8 同帧；失败则只 -10 |
| 物理失败 | `failure_suppresses_positive` | 真实 failure 帧（非超时） | failure_class、raw mask | -10 一次；无 +2/+8 |
| 超时 | `timeout_pays_failure` | 真实 hard timeout 帧 | timeout mask、done | -10 一次；无 +2/+8 |
| 同帧无正事件 | `no_positive_event` | 正常巡航一帧无事件 | flags 全 false | 无一次性项 |
| dt 一次 | `flat_cruise_full20`/`decel` | 同帧手算 | `0.02*sum` 对照 | 不是逐项 dt、不是 0.04 |
| 失败优先 | `failure_suppresses_positive` | 同帧 top/success/failure | 事件优先级 | failure 赢 |
| v07 差异 | 五个 `expected_v07` | 同一状态 | S2/v07 两个 total | 证明旧链不同 |
| 平地锚定 | `flat_anchor_examples` | 普通平地×名义×无噪声/延迟/外力 | teacher/student mu、clean mask、n_clean、grad | `L_anchor` 与梯度正确 |
| 动作对齐 | 无（需真实） | 同帧 `a_request/a_applied/c_exec` | 三数组 + frame | `a_applied=pd_target`；c_exec 单调；a_request 独立 |
| reset | 无（需真实） | 跨 auto-reset 两 episode | 状态锁存前后 | 仅 reset env 清零 |
| PPO 存储 | 无（需真实） | 两连续 sampling block | `(env_id,step)`、old/new logp、adv/ret、flatten | 唯一标签一一对应 |
| 异常结算 | 无（需真实） | reset/step/update/checkpoint 注入 | reserve/commit/release | actual 与物理步一致 |

## 4. 动作与命令对齐方案

### 4.1 字段定义

| 字段 | 来源 | 含义 |
|---|---|---|
| `a_request` | `env.action_manager.action` | 策略网络请求动作（raw） |
| `a_applied` / `q_target` | `env.action_term._processed_actions` | 实际 PD 目标（scale/clip/decimation 后） |
| `pd_target` | 同上 | 执行侧目标；必须与 `a_applied` 逐元素相同 |
| `c_request` | 计划/课程请求速度 | 应保持原始请求，不随 stop 衰减 |
| `c_exec` | S2 命令状态机 | 实际送入 actor 的延迟/ramp 后命令 |

### 4.2 同帧核对

在真实训练 env 上保存每个 `(env_id, step)`：

```text
a_request[t], a_applied[t], pd_target[t], q_target[t],
c_request[t], c_exec[t], phase[t], g_level[t],
reward_raw[t][20], reward_total[t], done[t], terminated/truncated[failure_class]
```

判据：

1. `a_applied == pd_target`（逐元素，float32 精确或同一 tensor clone）。
2. `c_exec` 在 DECEL 中单调下降，HOLD 为 0；`c_request` 不随 `c_exec` 下降。
3. `reward` 的 `c_exec` 与下一步 actor 实际消费的 `command` 同源；不能出现 `sample_command` 被上一步覆盖导致的双重 ramp。
4. 不需要两条独立 GPU 轨迹逐位相同；同一状态/动作/历史/命令/地形/终止标记下，逐位或容差内一致即可。

### 4.3 当前链的额外风险

`S2TrainEnv._s2_reward` 仍从 `env.sample_command` 取 `speed`，而 `_s2_task_state` 下一步把 `sample_command=env.command.clone()`，`env.command` 又被 adapter 在 reward 末尾写为 `c_exec`（`s2_train_real.py:319`、`:411`）。这会造成：

- `c_request` 在 stop 时变成上一帧的 `c_exec`；
- ramp 速度逐帧乘 `(1 - elapsed/100)`，形成二次衰减；
- 动作对齐出现“请求/执行/奖励命令”三者不同源。

验收必须专门构造一个 ramp stop 序列，记录 `c_request` 是否保持计划速度；若仍漂移，判 **FAIL**。

## 5. §5.4 平地锚定专项

### 5.1 独立手算例

`numerical_acceptance_handcalc_v13.json:flat_anchor_examples` 给出：

- 单个 clean 样本，`mu_s=0.1`（12 维），`mu_t=0`：`sum diff²/12=0.01`，`mean_clean=0.01`，`L_anchor=0.01`，`L=PPO+0.1*0.01`。
- 两个 clean 样本，差 0.1/0.2：`mean=0.025`，`L_anchor=0.025`。
- 无 clean 样本：`L_anchor=0`。

### 5.2 真实链要求

1. `S2TrainEnv` 或 `S2Trainer` 必须暴露逐环境 clean flags：`ordinary_flat`、`nominal_dynamics`、`no_noise`、`no_delay`、`no_force`；`clean_flat_mask` 由 `s2_loss.clean_flat_mask` 计算，不得用 task 前缀。
2. `role_plan` 必须包含 `envs`（或等价逐环境表）；否则 `clean_env` 全 false，锚定恒 0。
3. 教师必须是 `Frozen6400Teacher`（原始 48×32/旧图），不能是 `V09ActorCritic.migrate_6400_checkpoint` 的 40×30 迁移模型；当前 CLI 已把 teacher 设为 None，必须先实现旧图 teacher observation 适配，再重新打开锚定。
4. 聚合必须按 SDS:690-695：微批累计分子，最后统一除以整个逻辑 minibatch 的 `n_clean`；`n_clean=0` 时锚定为 0；`beta_flat=0.1`。
5. 测试必须对比“有无锚定”的梯度差，并证明 clean=0 时无锚定梯度；不能只断言 `teacher.calls>0`。

当前代码：**FAIL**（`role_plan` 无 `envs`、teacher=None 导致锚定直接禁用、clean 用 task 前缀、聚合按 micro 除全局 n_clean；见 §1.4）。

## 6. 实现方必须提供的 run / 证据（签字条件）

以下全部满足才可签“V13 数值验收通过”；缺一项即拒绝 r2 启动。

### 6.1 CPU 先做的证据（不占 GPU）

1. **手算对照包**：`numerical_acceptance_handcalc_v13.json` 的 13 个用例逐项对照；生产实现输出 raw terms/continuous/forward/events/total；按 §2.2 容差和离散精确比较。附独立复算脚本或人工表，禁止调用生产奖励函数生成 expected。
2. **PPO 存储单元**：用唯一 `(env_id,step)` 标签构造 fake transitions，核对 `flat_obs/flat_actions/flat_old/flat_adv/flat_ret` 同序；核对 `logical_minibatch_plan`/Adam 次数（`EPOCHS*MINIBATCHES`）。
3. **GAE 边界单元**：最后一步未终止（`last_value=V(next)`）与终止（0）两种；中途 done 后不跨 reset；`terminated/truncated/failure_class` 逐 env mask。
4. **异常结算单元**：reset 抛、step 抛、update 抛、checkpoint 抛四种；断言 ledger reserve/commit/release 与 `actual`；`_rollout_steps` 在 reset 前已初始化为 0。
5. **r1 拒绝链单元**：四个 SHA 在 gate、resume、attempt/output、ledger 指针被拒绝；r2 默认 attempt/output 不是 r1；无 r1 派生 checkpoint 被加载。
6. **§5.4 单元**：clean 0/1/2 例、teacher 旧图、`L_anchor` 手算、梯度方向；clean=0 时锚定梯度为 0。

### 6.2 真实后端 run（GPU；签字前至少一次）

**Run A：两连续 sampling block shadow**

- 正式 CLI、真实后端、非 fake、`num_envs=512`、`course=S2.1`、固定 seed；`--max-rollouts 2`（2×48=96 步，两个连续采样块）；独立 attempt/output（建议 `s2.1-v13-shadow` / `logs/a2_v09/s2/v13_shadow`），不得写 r2 正式目录、不得混入 r1。
- 使用隔离 ledger 副本或专用测试 obligation；不得在正式 `learning_S2_1` 上留下试跑消耗；如必须用正式台账，需先得到预算批准。
- 保存逐帧：`env_id, step, obs_keys, a_request, a_applied, pd_target, c_request, c_exec, phase, g_level, reward_raw[20], continuous, forward, top/success/failure/timeout, terminated/truncated masks, done, episode_length_buf, reset_mask, last_value, old_logp, value, advantage, return, minibatch_id, adam_step`。
- 至少一个环境跨 auto-reset；证明 `(env_id,step)` 对齐、终止不跨 reset、未 reset 环境不受影响。

**Run B：真实代表样本**

- 从 Run A/正式环境记录：`flat_walk` 巡航、`flat_stop` DECEL/HOLD、`stairs_walk` g_level t0/t0+1/t0+24/t0+25、`platform_walk` 直启、`legacy_l40` 同帧 top+success、物理 failure、hard timeout。
- 每个样本给出原始测量与手算 JSON 对照；不得用生产奖励函数生成 expected。
- 稀有边界可使用“真实状态回放/可控帧”，但必须有至少一个真实路径样本覆盖每类。

**Run C：动作对齐**

- 同一 stop 序列记录 `a_request/a_applied/pd_target/c_request/c_exec`；检查 `a_applied==pd_target`、`c_exec` 单调、`c_request` 不变、奖励命令与 actor 下一步命令同源。
- 不需要两条 GPU 轨迹逐位相同；同一状态/动作的逐位或容差一致即可。

**Run D：§5.4 锚定**

- 先实现 `Frozen6400Teacher`（48×32/旧图）→学生 40×30 的观测适配；当前 CLI 仍是 `teacher=None`，锚定禁用；普通平地×名义×无噪声/延迟/外力；记录 `mu_s/mu_t/clean/n_clean/L_anchor/beta_flat` 和锚定梯度。
- 证明 `L_anchor` 与 JSON 手算一致；`n_clean=0` 时梯度为 0；不包含楼顶/扰动样本。

**Run E：资源与预算**

- 修复后的实际执行栈复核 512 资源（显存/速度）与工程准入；试跑不计入 r2；记录实际 samples、ledger 事件、剩余额度。
- 账本：`learning_S2_1` 累计 20,299,776 + r2 有效学习 ≤ 800,000,000；r2 31,726 完整 rollouts = 779,698,176，合计 799,997,952（余 2,048）；不得给 r2 重开 8 亿。

### 6.3 签字判定

- 手算 13 用例：全部 term/total 在容差内，事件精确；任一失败 → 不签字。
- PPO/GAE/异常/r1 拒绝：全部通过；任一失败 → 不签字。
- 真实 Run A-E：两连续块、代表样本、动作对齐、锚定、预算全部通过；任一失败 → 不签字。
- 签字后仍需代码提交、来源 SHA、admission/probe 有效；V13 设计本身不构成启动授权。

## 7. r1 作废登记与 r2 拒绝链审计

### 7.1 已完整登记

`docs/v09_s2_1_r1_invalidated.md:4-17` 已登记：

| 检查点 | 样本 | SHA256 | 状态 |
|---|---:|---|---|
| `s2_1_000005013504.pt` | 5,013,504 | `59cf153d…` | 作废 |
| `s2_1_000010027008.pt` | 10,027,008 | `429384de…` | 作废 |
| `s2_1_000015040512.pt` | 15,040,512 | `247cc0c4…` | 作废 |
| `s2_1_000020054016.pt` | 20,054,016 | `16a06ea3…` | 作废 |

失效原因（v07 17 项，非 SDS §5）已写；样本数和 cursor 可重算；四文件保留原字节。**这四项完整。**

### 7.2 仍缺的登记字段

Q035 §5 要求“保留 … 代码/错误奖励身份、采样区间、失效原因”：

1. **采样区间**：应写 `[0, 20,299,776)` 样本 / `[0, 825]` rollouts（含 Q035 的 24,576 在途保守结算）；当前只写各 checkpoint 样本点，没有起止区间。
2. **代码/错误奖励身份**：应写 `s2_train.py` `41d56278…`、`s2_train_real.py`（r1 基线 `0d4a36fd…`）、v07 `REWARDS` 17 项、`success +10/timeout 0`；当前只写“旧 17 项”文字，没有可重算哈希。
3. **派生链范围**：应明确“r1 四 checkpoint + 任何基于它们的 resume/门禁/终验/晋级/导出/report 全部无效”；当前只写“不得用于门禁/终验/选模”，没有派生报告清单/拒绝测试。
4. **代码级拒绝**：`docs/v09_s2_1_r1_invalidated.md:17` 声明 gate 不得接受，但 `scripts/run_a2_v09_s2_gate.py:36-44` 仍只检查 `course`；四个 r1 checkpoint 的 `course=S2.1`，代码会接受。
5. **r2 拒绝链**：`scripts/train_a2_v09_s2.py:48` 默认 `--attempt-id s2.1-r1`，`:136` 默认输出 `train/<attempt>`；`S2Trainer.resume`（`s2_train.py:334-345`）加载任意 checkpoint，无 r1 SHA 拒绝。当前只有 `docs/v09_s2_1_r2_execution_contract.md` 的文字要求，没有代码/test 强制。

### 7.3 台账累计

当前 ledger 追加 Q035 在途保守结算后：

- 事件数 2,278；`learning_S2_1` reserve 827、commit 826、release 1；
- spent **20,299,776** / 800,000,000；reserved **0**；
- r2 可用 779,700,224；31,726 完整 rollouts ×24,576 = 779,698,176；累计 799,997,952（余 2,048）。

**确认**：账本累计与 r2 预算自洽；不得再给 r2 重开 8 亿。

## 8. 结论

- V13 的**公式手算参照已就绪**（JSON + 13 用例 + g_level 序列 + flat anchor 手算），可作为独立判据。
- 当前真实链仍有阻断：§5.4 锚定未真实生效/教师错误、terminated/truncated 未进 GAE、PPO `(env_id,step)` 未完整验证、异常结算 reset 前计数风险、命令链 request/exec 漂移、S2 适配器状态未全清、r1 未被代码级拒绝。
- 因此 **V13 暂不签字，r2 不得启动**；实现方按 §6 提供证据后，由独立复核逐项签字。
