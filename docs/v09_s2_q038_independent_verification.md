# DSH-Q038 独立验收：r2 算法偏差、§8 门禁稽核、OOM 取证

- 验收角色：独立验证代理（只读），不修改生产代码/测试/SDS/台账，不运行 GPU/mjlab。
- 日期：2026-11-24（会话内实时核对）。
- 结论：**不签字**。HEAD `bfb2550` 的三项算法偏差全部实测存在；§8 所述 1/2/3 亿门禁**从未执行**（缺测，非“未通过”）；OOM 主因已量化定位为采样阶段保留的 autograd 图（约 3.98 GiB/rollout），而非纯碎片化；总账算术自洽但“总 spent”的口径需更正。

## 1. 审计基线

| 项 | 值 | 来源 |
|---|---|---|
| 被审代码提交 | `bfb2550b88fb59b4c6b3f106a0b1dbae81795a8` | `git rev-parse bfb2550` |
| 当前 HEAD | `b72dea1`（仅文档；`git diff --stat bfb2550 HEAD` 只有 `doc/`、`docs/`、`logs/codex_changes.md`） | `git diff --stat bfb2550 HEAD` |
| `s2_train.py`（bfb2550） | `3a01e18985390027bda124945d9562eb2feed88acff7caed2c830338bdedff68` | 快照 `logs/a2_v09/s2/q038_verify/snapshot_bfb2550/` |
| 与 r2 checkpoint 内 `code_sha256` | 一致（`3a01e189…`） | `s2_1_000100270080.pt` metadata |
| SDS | v1.2 `09fa2d4cdd858914c46a51fd52670c20cb20f2b6f3e76e6d266054574f64ccbe` | `doc/sds/SDS_A2_v0.9_Sequential_Map_Walk_Stop_Robustness.md` |
| 审计时工作树 | **脏**：`src/tasks/velocity/v09/s2_train.py` 已被实现代理改为 `2452d96d…`（WIP，未提交，见 §8） | `sha256sum` + `git status` |

> 说明：本报告全部 `file:line` 均指 **bfb2550 快照**（`logs/a2_v09/s2/q038_verify/snapshot_bfb2550/*.py`，行号与 bfb2550 原文一致）。工作树正在被实现代理修改，不能作为审计对象。

## 2. 三项偏差确认（全部仍在 HEAD 中）

### 偏差 1：GAE 后 raw advantage 直接进 policy loss，缺“整个 rollout 一次归一化”

- `s2_train.py:368` `flat_adv = advantages.reshape(-1)`；`s2_train.py:249-250` 的 `adv_t/ret_t` 就是 `s2_runner.gae` 的原始输出。
- `s2_train.py` 全文**没有**任何 `normalize_advantage` 调用（`grep -n normalize s2_train.py` 只命中 normalizer 冻结相关行）。
- 对照语义：v07 `ppo.py:521` 在 GAE 后立即 `normalize_advantage(advantage)`；`ppo.py:181-184` 为 `(a-mean)/(std+1e-8)`；SDS §2.4“advantage 在完整 rollout 上一次归一化”。
- returns 本身正确：`s2_runner.py:105-139` 返回 `advantages + v`（原始），未被归一化/缩放，符合 Q038 B.1。
- 固定批实测影响：raw adv std=1.0133，归一化后 mean=0、std=1；若实测 advantage 尺度偏离 1，policy 梯度与 entropy/std 的相对权重随之漂移，等于隐式改配方。

### 偏差 2：value loss 只有 MSE，缺既定 clipped value loss

- `s2_train.py:407` `value_loss = (value - flat_ret[micro]).pow(2).mean()`。
- 仓库内已有的 clipped 版本 `s2_runner.py:92-104`（`old + clamp(v-old,±clip)` 后取 max）**从未被调用**（`grep -n "value_loss(" s2_train.py` 无命中）。
- 既定约定：v07 `ppo.py:280-285`（同一 old-value clamp，`clipped_value_loss=True`），SDS §2.4 表“clipped value loss 开启”。
- 判别例（本次实测，`reference_expected.json.value_loss_discrimination`）：`v=1.15, v_old=0, ret=1.0` → clipped=0.6400，MSE=0.0225（28.4×）。该配置下两式不可互换。

### 偏差 3：KL 在逻辑 minibatch 循环外、每 epoch 用整 rollout 算一次并调 LR

- `s2_train.py:351` 进入 epoch 循环；`s2_train.py:372-378` 对**全部 24,576 样本**分块前向（`no_grad`，仅在内存上分块，语义仍是整 rollout）；`s2_train.py:379-381` 用整 rollout 算 KL 并 `adjust_learning_rate` 一次；`s2_train.py:382-383` 写回 lr；`s2_train.py:388-441` 再跑 4 个逻辑 minibatch、各自 Adam。
- 结果：**每个 rollout 只有 5 次 LR 决策**（而非 20），且同一 epoch 内 4 次 Adam 共用同一个 lr；KL 是在该 epoch 任何一次更新之前、对混合样本测得的，不是“该逻辑 minibatch”的 KL。
- 对照语义：v07 `ppo.py:258-271`（每 epoch `randperm` → 每 minibatch 先算 KL→调 lr→再反传→Adam）；SDS §2.4“逻辑 minibatch 内先按全体样本计算 KL 并决定本次学习率，再计算微批梯度”。
- Adam 次数**正确**：`s2_train.py:440` 处 5×4=20 次；r2 日志 `anchor.observations` 在 rollout 12,912 处为 258,240 = 20×12,912，实证每 rollout 20 次逻辑更新。

## 3. 其他核对项（bootstrap / 对齐 / 锚定 / 快照）

| 核对项 | 判定 | 证据 |
|---|---|---|
| time-major 对齐 | PASS | 采样为 step 前 obs（`s2_train.py:174`），reward/value/mask `stack(...,1)`（`:213-226`），obs `cat` 于 dim0（`:355`）、action `stack` 后 reshape（`:366`）、old logp `stack` 后 reshape（`:45-46,333`）、adv `[T,N]` reshape（`:368`），全部为 `(t, env)` 序 |
| bootstrap / reset | PASS（按 Q035 合同），与 v07 谱系不同（已文档化） | 内部 terminated/truncated/failure/timeout 一律 0 bootstrap（`:227-244`, `s2_runner.py:105-139`）；仅采样边界且末步未 done 时用 `V(next_obs)`；`next_values` 在 `no_grad` 下（`:232-234`）。v07 `ppo.py:151-177` 对 truncation 用 terminal obs 值 bootstrap——差异见 `v09_execution_rulings.md:357` 的 Q035 裁定，非静默改动 |
| mask 缺失回退 | FAIL（健壮性） | `s2_train.py:216-226`：任一行缺 mask 就整体回退 `zeros_like(done)` → GAE 会跨 reset 传播而**不报错**；应改为抛异常 |
| §5.4 锚定分母 | PASS | `s2_train.py:392` 取逻辑 minibatch 的 `n_clean`，`:431-433` 统一除 `12*n_clean`，与 `s2_loss.py:35-56` 的 `AnchorAccumulator` 一致；本次用独立参照与仓库 helper 双验（`reference_selfcheck.json.repo_helpers.anchor_acc_matches=true`）。错误做法的偏差量：同一单块固定批上“除全部样本”得 3.2639e-4，比正确的 4.8958e-4 小 33.3%；不等长微批固定批上“各微批 mean 再平均”比正确值大 16.9 倍（§4） |
| 锚定 teacher 状态对齐 | **FAIL** | `s2_train.py:200` 在 `self.env.step()`（`:179`）**之后**取 `info['teacher_observation']`；`s2_train_real.py:727` 的该字段由 `teacher_observation()`（`:624-631`）从 step 后的 robot 数据生成，而 student 样本用的是 step 前 obs（`:174`）。锚定因此把 `mu_s(obs_t)` 与 `mu_t(obs_{t+1})` 配对（t 为末步时甚至是 reset 后状态）。需改到 step 前采 teacher obs 并加 1 步位移测试 |
| teacher 缺失静默回退 | FAIL（健壮性） | `s2_train.py:422-423`：`flat_teacher` 为空时 `teacher_item = clean_item`，锚定退化为“学生对自己”且不报错 |
| clean-flat 口径 | 观察项 | `s2_train_real.py:607-615` 只把 `flat_walk` 记为 ordinary_flat，`flat_stop` 环境永不进入 §5.4；r2 日志每逻辑 minibatch `n_clean=1236`（6,144 的 20.1%，对应整 rollout 4,944/24,576）。§5.4 未明确 flat_stop 是否算“普通平地”，需规划裁定 |
| 采样快照 vs env 缓冲 | PASS | v07 `env.py:631-635` 每步 `clone()` 返回新 tensor；`s2_train.py:183` 每步新建 dict；teacher 切片亦为当步新张量。未发现批内张量随 env 缓冲变动 |
| 采样是否保图（OOM 源） | **FAIL** | `s2_train.py:174-178` 的 `action_distribution/evaluate` 未用 `no_grad`，`log_probs/values` 带 graph 存入 `:183-184`，再被 `:333` stack；全 rollout 期间不释放（量化见 §6） |
| minibatch 是否打乱 | 观察项 | `s2_train.py:386` `order = torch.arange(...)` 每个 epoch 相同分区；v07 `ppo.py:258` 每 epoch `randperm`。SDS §2.4 未强制打乱，但 5 个 epoch 用同一批内顺序会降低更新多样性，应在实现说明中记录 |
| 采样批是否被改写 | PASS | 未发现对 `transitions`/`flat_obs` 的写操作；`_clean_env_cache` 只读缓存（`:140-150`），角色固定时不会陈旧 |

## 4. 独立 PPO 参照（CPU）与固定批

产物：`logs/a2_v09/s2/q038_verify/`

| 文件 | SHA256 | 作用 |
|---|---|---|
| `reference_ppo.py` | `72c7980989b314ad5c713430d020ff509517a0b8a81c622e158d80d09a573f5f` | 从零实现 §2.4（正确语义）与 bfb2550 语义（旧版）两套更新 |
| `reference_expected.json` | `fb6fc4f15141a4fee73bf4b5f20881666d52cf3e9234d887386d42ff5298d1af` | 固定批期望值；`batch_sha256=482cdadb9db8ee2c6a02e0b54c6c176988a91b9ddcb5c4fc7356a14c9d056945` |
| `selfcheck_reference.py` / `reference_selfcheck.json` | `3e457583…` / `a2b2a95b…` | 手算 GAE + 第二个 torch-autograd 实现逐项对拍 + 仓库 helper 交叉验证 |
| `compare.py` | `96349527682cec7e81df137976564161cbcd19a944a98e78563bb6829a36a73e` | 判定实现方 fixed-batch 证据（格式见文件头 docstring，15 项检查） |
| `selftest_judge.py` / `selftest_judge.json` | `e0971b0f…` / `9b43f5bb…` | 判定器自测：正确证据 15 项全 PASS；旧语义证据 6 项 FAIL（负对照有效） |

固定批：T=2、N=4（8 样本，time-major），小解析模型 `mu=w_a·x+b_a`、`v=w_c·x+b_c`、`sigma=clamp(std)`；初始 `w_a=0.5,b_a=0.1,w_c=-0.4,b_c=0.2,std=0.30`。含 terminated@t0 env2、truncated@t1 env1、truncated@t0 env3、无 done env0（边界 bootstrap）。

独立参照关键期望值（float64；实现在 float32 下按 `atol=rtol=1e-5` 判定）：

```
raw adv (t0) [ 1.815933 -1.045450 -1.400000  0.150000]
        (t1) [ 0.446500 -0.900000  0.148500  0.900500]
returns (t0) [ 1.815933 -0.445450 -2.000000  0.250000]
        (t1) [ 0.846500 -1.000000  1.148500  0.700500]
normalized adv: mean=0, std=1（raw std=1.013269）
```

| Adam# | epoch/mb | LR（正确） | KL（正确） | policy | value（clipped） | ratio |
|---|---|---|---|---|---|---|
| 1 | 1/1 | 3e-05 | 0.0 | -0.365889 | 2.195290 | 1.000000 |
| 2 | 1/2 | 4.5e-05 | 4.985e-07 | 0.632106 | 0.991370 | 1.000731 |
| 3 | 1/3 | 6.75e-05 | 6.706e-09 | 0.238166 | 0.504747 | 1.000072 |
| 4 | 1/4 | 1e-04 | 6.532e-07 | -0.503113 | 0.416416 | 0.999081 |
| 5 | 2/1 | 1e-04 | 5.597e-09 | -0.366041 | 2.195148 | 1.000009 |
| … | | 1e-04 ×17 | 逐 minibatch 不同 | | | |

- 正确语义：20 次 LR 决策、序列 `3e-5, 4.5e-5, 6.75e-5, 1e-4×17`；旧语义：5 次 LR 决策、序列 `3e-5×4, 4.5e-5×4, 6.75e-5×4, 1e-4×8`（同 epoch 内 4 次 Adam 共用 lr，判定器据此判别）。
- 旧语义在固定批上另有两个可测征兆：`epoch 内 KL 恒定`（4 个 minibatch 同值）；`used_adv` 与 raw 相同（未归一化，平均绝对差 0.01550，尺度比 0.9869）。
- 锚定固定批（6×12，clean=[T,F,T,T,F,T]，n_clean=4）：numerator=0.0195833，`L_anchor=4.895833e-4`，`dL/dmu_s = β·(2/12)·(mu_s-mu_t)/n_clean`；错误分母对照（单块批）：除全部样本 3.263889e-4（比正确值小 33.3%）。不等长微批判别批（6+2 样本、n_clean 2+1）：正确 `β·Σnum/(12·n_clean)=1.058681e-4`；“先各微批取 mean 再平均”=1.792917e-3（**16.9×**）；“除全部样本”=4.764063e-4（4.5×）。判定器对应新增检查 `anchor_accumulates_microbatch_numerators`。
- 自证强度：手算 GAE 一致；解析实现 vs torch-autograd 最坏绝对差 policy 4.7e-10、value 1.4e-10、params 4.3e-11；`s2_runner.gae` / `normalize_advantage` / `s2_loss.AnchorAccumulator` 三者与参照一致。
- 判定器自测：`judge_is_discriminating=true`（正确 15/15 PASS；旧语义 FAIL：`advantage_normalized_once_over_rollout`、`kl_per_logical_minibatch_from_logp`、`lr_decision_before_each_adam_step`、`kl_differs_between_minibatches_in_an_epoch`、`value_loss_clip_convention_on_cases`、`anchor_accumulates_microbatch_numerators`）。

## 5. §8 门禁稽核（累计 1/2/3 亿）

真实台账：`logs/a2_v09/s2/ledger.jsonl`，SHA256 `c3a69536ae4f003b049f3de742a52f9d4c4c425d3b9f3d164b2f71924f81549c`，28,409 行，schema 11。

| 阈值（累计学习样本） | 精确跨越节点 | 已持久化候选（最近） | 候选文件 SHA256 / state SHA | 完整 S2.1 门禁 A/B 是否执行 | 实际评估样本 | 门禁义务台账 | 调度/冻结证据 |
|---|---|---|---|---|---|---|---|
| 1e8 | rollout 4,070 = 100,024,320（**未落盘**） | `s2_1_000100270080.pt`（4,080 = 100,270,080，晚 245,760 = 10 个 rollout） | `28c4a6f576b1793c37c692f73b4bb48c6b891b85193effcca3599cdd409e9fb6` / `79cacbbc1bc2…` | **否（missing measurement）** | 0 | `gate_admissions_up_to_40`：0 条 reserve/commit | 无 |
| 2e8 | rollout 8,139 = 200,024,064（**未落盘**） | `s2_1_000200540160.pt`（8,160 = 200,540,160，晚 516,096） | `b603d37e03856ae13c9555e008d7d4f00325ceda30ddf64594a03e73ddad0607` / `48180a859048…` | **否** | 0 | 同上：0 | 无 |
| 3e8 | rollout 12,208 = 300,023,808（**未落盘**） | `s2_1_000300810240.pt`（12,240 = 300,810,240，晚 786,432） | `7b053b8d6035c6cc128a0e0b8d9e15e8b5f317e3d6a2d816042b8cb75396d6a1` / `386623e91c84…` | **否** | 0 | 同上：0 | 无 |

补充事实：

1. **63 个 checkpoint 全部 `gates_passed=[]`、`gate_uses=0`、`gate_ab_sequence='gate_A'`**（`gate_audit.json.checkpoints`）；`s2_train.py:122-123` 初始化这两个计数器后**再无任何写入**（bfb2550 中 `gate_uses` 只出现于 `:123` 初始化与 `:497` 写元数据，`gates_passed` 只出现于 `:122` 与 `:493`）。即：训练侧根本没有到点门禁/冻结实现。
2. 仓库内**不存在**任何 `S2*_gate_result.json`，也不存在 S2 门禁输出目录（`find . -name "S2*_gate_result*.json"` 空；`logs/a2_v09/s2/gate_dry_run/**` 只有 plan，`physics_started=false, ledger_written=false`）。
3. 台账中的 `gate_admissions_up_to_40` 与 `final_sets_3_with_frozen_comparisons` **只出现在初始 `stage_reserve` 计划里**，没有任何 reserve/commit/release 事件 → 门禁与终验均 0 次、0 样本。
4. 现有评估产物**只有参考缓存**，且不完整：
   - L40 参考缓存仅 **A 套**：`gate_caches/gate_A/l40-r4/reference_cache.json`，`source_sha256=efa9caa8…`（s1_least），`settlement.actual=75,204`，`generated_at_utc=2026-11-23T02:55:00Z`，且 `salvaged_from_case_files=true`（由 case 文件拼装，不是完整原子跑批）。`l40-r2/r3` 只有 cases、无 cache；`gate_A/6400`（首发）只有 cases、无 cache。
   - 6400 平地参考缓存仅 **A 套**：`gate_caches/gate_A/6400-r2/reference_cache.json`，`source_sha256=30ed23ba…`、`model_type=original_6400`、`migrated=false`、`observation_shape=[3,48,32]`、`action_consistency.passed=true, max_abs_delta=0.0`；但其 `concurrency.concurrent_with_training=true`、`allow_concurrent=true`（训练进程 2,015,768 在跑，free 6,364.8 MiB ≥ 3,072）。门禁手册要求训练窗口内拒绝并发；虽记录齐全，仍属调度违规。
   - **B 套参考缓存完全缺失**（`gate_B/` 只有 `l40_gate_B_plan.json`；无 6400 B）。`l40_milestone_cache_gate_B` 台账 0 事件。
   - `original_6400_flat_gate_A(_rerun)` 已各结算 234,000；`l40_milestone_cache_gate_A(_rerun/_rerun2/_rerun3)` 结算 1,554 / 75,179 / 75,147 / 75,204。
5. **是否有任何门禁结果只用了参考缓存**：没有任何门禁结果；参考缓存只是未来门禁的输入。因此不构成“用缓存代替候选评估”，但也不构成任何测量。
6. **是否跳过晋级/冻结/回滚**：候选晋级、到点冻结、`gate_uses` 计数、§8.2 回退触发条件全部**没有发生**。§8.2 要求“连续两次正式门禁失败”才回退，由于 0 次门禁，r2 的行为退化（closure 记录 10.03M 起坍缩）从未进入门禁判定，也**不能**据此声称回退条件成立；这同时说明“到点冻结/门禁执行”链条完全缺失。
7. **录像/替代 checkpoint 不是门禁证据**：`logs/a2_v09/s2/videos/sweep_*.probe.json` 是 5M/10M/20M/50M/100M/200M/315M 的**单 env、600 步确定性**探针（`sweep_000100270080.mp4.probe.json`：`final_local_x=-0.7073, final_step=599`），不是 36 组×20 例 A/B，也未走 `run_a2_v09_s2_gate.py`；本次审计不作门禁证据。

## 6. OOM 取证与对账

崩溃点（`logs/a2_v09/s2/r2.log:13131`）：`Tried to allocate 20.00 MiB`，进程 5.73 GiB，PyTorch allocated 4.59 GiB，reserved-unused 856 MiB，free 86.56 MiB。traceback 落在 `s2_train.py:377`——**每 epoch 整 rollout KL 前向**（`r2.log` 未记录 epoch 序号；“epoch5”无日志依据，只能作为假设）。

CPU 实测保留量（`oom_forensics.py`，同形状/FP32；`oom_forensics.json`）：

| 来源 | 单步 | ×48 步/rollout | 说明 |
|---|---:|---:|---|
| actor 采样前向（带 graph） | 41.54 MiB | 1,994.1 MiB | `s2_train.py:174-178` 未 `no_grad` |
| critic evaluate（带 graph） | 41.40 MiB | 1,987.4 MiB | 同上 |
| 合计 autograd 图 | 82.95 MiB | **3,981.6 MiB** | 直到 `update()` 结束才随 `transitions` 释放 |
| obs 快照（3,908 float/样本） | — | 366.4 MiB | 学生 5 键 |
| teacher clean 快照（4,916 float/样本，103/512 env） | — | 92.7 MiB | `s2_train.py:200` 切片后 detach |
| 每 epoch `flat_obs/flat_teacher` 重建 | — | 459.1 MiB | `s2_train.py:355-361` 每 epoch 重新 `cat`，反复分配/释放 |
| trace `info`（全 512 env 的 teacher obs，仅采集期） | — | 460.9 MiB | `s2_train.py:205-210`，`--trace-dir` 打开时每步都存 |

- `update()` 期间同时存活的可避免集合 = 3,981.6 + 366.4 + 92.7 + 459.1 ≈ **4,899.8 MiB**，与日志“4.59 GiB allocated”同量级。采样图单项就是失败请求 20 MiB 的 **199×**、reserved-unused 856 MiB 的 **4.65×**。
- 结论：20 MiB 失败**不能**只用碎片化解释（856 MiB reserved-unused 只是共同结果）；保留的 autograd 图 + 每 epoch 重建是主因，符合 Q038 D.1 的第一顺位怀疑对象。录像、teacher 前向本身不是主因。
- 实现方内存验证必须给出：修正算法下 512 env × 48 步 × 5 epoch 连续 ≥5 个真实 rollout 的 `torch.cuda.max_memory_allocated`、`max_memory_reserved` 峰值，`mem_get_info` 的 free MiB 前后值，**无 OOM**；并覆盖最重场景（`--trace-dir` 开启、512 env）与一次 save/restore 恢复检查；同时给出修正前后同一固定批的数值/梯度对比（Q038 D.4）。预期修正后采样图应降为 0（`no_grad`）且 `flat_obs` 只建一次，保留量应显著低于本次实测的 4.90 GiB。
- `r2.log` 只记录 12,964 条 rollout JSON（最后 `rollouts=12964`），第 12,965 个 rollout 的样本已 `commit` 但 PPO 更新崩溃；不存在“epoch5 已部分 Adam 提交”的日志证据，D.5 要求的三游标（物理步/完整更新/持久 checkpoint）未在任何文件中分开记录。

### 预算对账（`audit_accounting.py` → `gate_audit.json.reconciliation`）

| 声明 | 实核 | 结论 |
|---|---|---|
| 总 338,927,616 | `learning_S2_1` 净 commit = 338,927,616（20,299,776 + 12,965×24,576） | 数值 PASS |
| r2 318,627,840 | 12,965 条 r2 commit，全部 24,576，和 = 318,627,840 | PASS |
| reserved=0 | `learning_S2_1` outstanding = reserve − commit − release = 0 | PASS |
| 余 461,072,384 = 18,761×24,576 + 2,048 | 800,000,000−338,927,616=461,072,384；18,761×24,576=461,070,336，余 2,048 | PASS |
| 可用学习样本 461,070,336 | 18,761 个完整 rollout | PASS |
| 计划 31,726 | (800,000,000−20,299,776)//24,576 = 31,726 | PASS（是 r2 起点余额，不是当前余额） |
| 12,965/31,726 = 40.87% | 0.4087 | 算术 PASS，但其中 1 个 rollout 只采集未训练（完整更新 12,964），Q038 A.6 的告诫成立 |
| “总账实核 338,927,616” | 这是 **S2.1 学习义务**，不是阶段总账；非学习净 commit = 5,758,114，阶段净 = **344,685,730** | 口径需更正（不涉及丢失/重复计费） |

## 7. 逐项判定（Q038 A/B/D）

| 项 | 判定 | 依据 |
|---|---|---|
| A.1 r2 关闭、保留、不晋级/导出/续训 | **PASS** | `docs/v09_s2_r2_closed_q038.md`；63 个 checkpoint 保留；台账无导出/续训事件；仅有诊断录像 |
| A.2 OOM 为中断、N10 退化早于 OOM | **PASS**（作为描述） | closure 与录像探针一致；本次 OOM 机制见 §6 |
| A.3 授权范围 | **PASS** | 本次只做只读取证；工作树改动来自实现代理，非本会话 |
| A.4 未来 attempt 以 `s1_least.pt` 为源、新 Adam、限额试训待裁定 | **NOT-COVERED** | 尚无新 attempt 的入口/Adam/限额冻结文件 |
| A.5 总账实核 | **PASS（算术）/ 口径需更正** | 见 §6 对账；338,927,616 仅为 S2.1 学习义务 |
| A.6 40.87% 不得视为成功更新 | **PASS** | 12,964 完整更新 vs 12,965 计费 rollout |
| B.1 全 rollout 一次 advantage 归一化 | **FAIL** | `s2_train.py:368`，无 normalize 调用 |
| B.2 clipped value loss | **FAIL** | `s2_train.py:407`；`s2_runner.py:92-104` 未被调用 |
| B.3 每逻辑 minibatch KL/LR | **FAIL** | `s2_train.py:372-383`；5 次而非 20 次 LR 决策 |
| B.4 固定批三方对拍证据 | **NOT-COVERED** | 实现方无任何 fixed-batch 证据；独立参照已就绪（§4），可直接判定 |
| B.6/B.7 诊断、轨迹、20 项对比 | **NOT-COVERED** | 本次任务不含；未审 |
| D.1 OOM 取证 | **PASS（本报告完成）** | §6，保留量量化到 4.90 GiB |
| D.2 no_grad 采样/不可变快照/分块 KL/锚定同分母 | **FAIL** | `s2_train.py:174-178,183-184,333`；KL 仅“分块”未改语义；锚定跨 micro 持图（`:424-433`） |
| D.3 512×48、FP32、20 次 Adam、归一化/clip/聚合不变 | **PASS（现状）** | 配置 `s2_runner.py:1-31`；无 autocast/amp；20 次 Adam 实测（§2 偏差 3） |
| D.4 修正后真实 512 五 epoch 显存复核 | **NOT-COVERED** | 无峰值/余量/无 OOM 证据 |
| D.5 崩溃轮部分提交与三游标 | **NOT-COVERED** | 仅台账 24,576 一条 commit；无物理步/完整更新/持久游标分离记录，无“废弃部分更新”的书面记录 |
| D.6 由 DSH 协调独立验收 | **进行中（本报告）** | 本报告即该项输入；尚不能签 |

**阻碍限额试训（5M→10M）的项**：B.1/B.2/B.3 修复未提交、B.4 证据缺失、锚定 teacher 错位、D.2/D.4/D.5 未完成、§8 到点门禁与冻结机制未实现（63 个 checkpoint 全 `gate_uses=0`）、B 套参考缓存缺失。在这些补齐前，任何 5M/10M 试训都无法证明用的是已修正算法，也无法在阈值处留下可稽核门禁。

## 8. 实现方签字前必须补齐的证据（精确清单）

1. **提交**：一个新 commit（附 SHA），其 `s2_train.py` 满足：whole-rollout 一次归一化、returns 仍为 raw adv+V_old、clipped value loss（old-value clamp）、每逻辑 minibatch 先 KL 后 LR 再 Adam（20 次 LR 决策）、采样 `no_grad` + detach 快照、teacher obs 与 student 同一步、缺失 mask/teacher 时抛异常而非静默回退、去重后的每一步诊断字段。附对应单元测试与全量回归结果。
2. **固定批证据**（新文件，`q038_fixed_batch_evidence_v1` 格式，见 `compare.py` docstring）：至少 1 份“真实存档入模批”+1 份本报告冻结批；由 `compare.py` 判定 **PASS**（全部检查项：冻结批模式 15 项 / 自定义批 13 项，含 `value_loss_cases` 判别三元组与 `anchor_microbatches`）。
3. **锚定对齐证据**：证明 `mu_teacher` 与 `mu_student` 取自同一状态（t 而非 t+1），含 1 步位移的失败用例与 `n_clean` 兼容性说明。
4. **显存证据**：修正后 512×48×5 epoch 连续 ≥5 rollout 的 `max_memory_allocated/max_memory_reserved` 峰值、`mem_get_info` free MiB、无 OOM；最重场景（trace 开启）+ 一次 save/restore；以及修正前后保留量对比（本报告可作基线：采样图 3,981.6 MiB、obs 366.4 MiB、teacher 92.7 MiB、每 epoch 重建 459.1 MiB）。
5. **台账/游标**：崩溃轮的三游标记录与“废弃部分更新”书面结论；新 attempt 的原子预留（含门禁/缓存/探针上界）与 attempt-id；`learning_S2_1` 口径更正文案（或确认仅指 S2.1 学习义务）。
6. **门禁整改**：到点调度器（阈值→完整 rollout 边界→冻结候选→A/B→`gate_uses`/`gates_passed` 落盘→待处理节点阻止继续训练）；在 1/2/3 亿节点保存候选（或按 Q038 记录“精确节点未落盘”并裁定）；补采 L40/6400 的 **B 套**参考缓存（新窗口、串行、无并发）与 A 套重跑（当前 L40 A 为 salvage、6400 A 为并发采集）；随后用同一冻结候选 SHA 跑 A/B，登记实际评估样本与 `gate_admissions_up_to_40` 事件。若本轮不补，则按 Q038 §8 将 1/2/3 亿明确登记为**缺测**并请求规划裁定。

## 9. 复现命令与产物

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
/home/xxs/anaconda3/envs/a2/bin/python logs/a2_v09/s2/q038_verify/reference_ppo.py
/home/xxs/anaconda3/envs/a2/bin/python logs/a2_v09/s2/q038_verify/selfcheck_reference.py
/home/xxs/anaconda3/envs/a2/bin/python logs/a2_v09/s2/q038_verify/selftest_judge.py
/home/xxs/anaconda3/envs/a2/bin/python logs/a2_v09/s2/q038_verify/oom_forensics.py
/home/xxs/anaconda3/envs/a2/bin/python logs/a2_v09/s2/q038_verify/audit_accounting.py
# 判定实现方证据：
/home/xxs/anaconda3/envs/a2/bin/python logs/a2_v09/s2/q038_verify/compare.py \
  --evidence <implementation_evidence.json> --out <judgement.json>
```

- `logs/a2_v09/s2/q038_verify/selftest_judge.json`：`judge_is_discriminating=true`（正确 15/15 PASS，旧语义 6 项 FAIL）。
- `logs/a2_v09/s2/q038_verify/reference_selfcheck.json`：`hand_gae.reference_matches_hand=true`；解析 vs torch 最坏差 policy 4.70e-10 / value 1.40e-10 / params 4.31e-11；`repo_helpers.gae_matches_s2_runner/normalize_matches_s2_runner/anchor_acc_matches` 均 true。
- 台账 SHA、缓存/收据清单、门禁判定明细：`logs/a2_v09/s2/q038_verify/gate_audit.json`。
- bfb2550 源码快照（行号基准）：`logs/a2_v09/s2/q038_verify/snapshot_bfb2550/`。

## 10. 范围声明

- 本会话未修改任何生产代码/测试/SDS/台账；未运行 GPU、mjlab、训练或评估。
- 审计期间实现代理正在改 `src/tasks/velocity/v09/s2_train.py`（工作树 `2452d96d…`，含 `normalize_rollout_advantages`、`clipped_value_loss`、minibatch 内 KL 的雏形）；该 WIP 未被本报告审计，也不能作为签字依据，需其提交并附 §8 证据后重新判定。
