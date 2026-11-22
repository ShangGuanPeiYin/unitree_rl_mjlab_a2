# A2 v0.9 S2.1～S2.3 训练栈独立验收报告（V6，含 Q028/Q029 与 V5 修复）

- 审核 agent：A2-S2-VERIFY（独立验收子 agent，未参与实现）
- 审核时间：2026-11-22（快照 14:44–14:52）
- 仓库/分支/HEAD：`/home/xxs/dog/unitree_rl_mjlab_a2` / `main` / `b609636`
- 交付 commit：T3 `400cac7`；其后 `b609636`（V5 修复）；工作区在 `s2_budget.py`、`s2_ledger.py`、`tests/test_v09_s2_0_ops.py` 仍有未提交改动（审核按当前工作区）。
- 测试基线：`pytest tests/test_v09_s2_*.py` → **144 passed**。
- 范围：纯 CPU；未跑物理/训练；未改实现/测试/SDS/台账、未 git 写。
- 机器可读件：`logs/a2_v09/s2_0/verification/train_stack_audit_20261122_r6.json`

## 0. 结论总表

| # | 项 | 结论 | 关键证据 |
|---|---|---|---|
| 1 | rewards §5.1/§5.3 | **阻断 1 项** | 20 项权重全对；`stable_stand` 在 DECEL 且 c_exec=0.25 时仍付 +1 |
| 2 | loss §5.4 | **通过** | 微批分子相加/统一除 n_clean；n_clean=0→0；β=0.1；clean mask 正确 |
| 3 | runner §2.4 | **阻断（不完整）** | 参数/LR/std/GAE 正确；缺 entropy 项、KL 计算、每逻辑 minibatch 的 LR→Adam 接线、教师/normalizer 冻结 |
| 4 | resources §2.3 | **阻断 1 项** | 门槛/计费/顺序意图在；512 可单独被选，无需 1024 失败证据 |
| 5 | checkpoint §10.1 | **阻断（不完整）** | 字段齐全；无 loader/跨版本守卫，部分字段仅存在性检查 |
| 6 | ledger Q021 | **阻断 1 项** | 命名空间/cap/预留非消费/不复用 S1 均对；operation_id 无去重 |
| 7 | acceptance §6.5 | **阻断（多项）** | 阈值在；缺站立 roll/pitch/高度、无巡航窗拒绝、27×100 组成、6400 平地保持 |
| 8 | train CLI | **通过（附策略缺口）** | 1024 默认、非 dry-run 拒绝；可直接 `--num-envs 512` |
| 9 | Q027/V5 修复 | **通过** | 同帧同 g、快照版本、旧口径报错；H 走 v07、dry-run 30/20、docstring、case_record 配置 SHA 已修 |
| 10 | Q028/Q029 | **通过向量；2 项缺口** | 强制向量 0/0/0.02/0.50、Σg=13、0.5、25/26、0.02/0.50 完全一致；首米窗未实现、首次满值区间起点口径错 |

## 1. s2_rewards（§5.1/§5.3）

- 20 项权重逐项与 §5.3 两列一致（pitch `-5·g·gx²`、超速 `-2·g·max(v−c_exec−0.15,0)²`、roll −5、vz −1、height −0.1、action −0.01、default −0.05、dof_acc −2.5e-7、dof_vel −5e-4、ang_xy −0.05/−0.20、air 0.2/0、slip −0.15、clearance −0.5、joint −0.1、center/heading −0.5、limits −10、stable 0/+1）。
- 连续项只乘一次 0.02；前进 `0.4·Δx` 仅 `I_cruise`；`I_cruise` 三条件（c_request>0、非 DECEL/HOLD、c_exec 达目标）；停车关闭前进与腾空并返回清累计标志；抬脚目标 ground/platform=0.075、tread=0.22。
- 失败/超时同帧：+2/+8 均被抑制、−10 只付一次（实测 total 仅 −10 项）；成功 +8、登顶 +2 由 `*_once` 标志控制。
- **B-V6-1（阻断）**：`stable_stand` 项在 `phase=DECEL, c_exec=0.25` 时仍加权 +1（实测 `weighted_terms.stable_stand=1.0`）。§5.3 要求 `I_stable×I_abs(c_exec)≤1e-7`，即只在零指令时支付；DECEL 期间不得给 +1。

## 2. s2_loss（§5.4）

- `clean_flat_mask` 同时要求 ordinary_flat、nominal、no_noise、no_delay、no_force；反例（楼顶/扰动）由调用方 flags 排除。
- 微批只累计 `Σ_j diff²/12` 分子与 clean 样本数；最终 `numerator/n_clean`；实测两微批 1.0+5.0 → 5.0，与精确值一致；`n_clean=0` → 0.0；β_flat=0.1。
- teacher=6400 旧图：模块只接收 `mu_teacher`，不校验来源；由 runner/checkpoint 负责（见 §5 缺口）。

## 3. s2_runner（§2.4/GAE）

- PPOConfig 与 §2.4 一致：gamma .99、λ .95、clip .2、value 1、entropy .003、betas .9/.999、eps 1e-8、wd 0、LR 3e-5、上下限、KL 阈值、std 0.05–1、grad 1.0。
- `adjust_learning_rate`：KL>0.02 ÷1.5、0<KL<0.005 ×1.5、夹在 [1e-5,1e-4]；`project_std` 夹 [0.05,1.0]；`weighted_accumulate` 按样本数加权、一次平均；`logical_minibatch_plan` 4 个逻辑 minibatch、微批 ≤512、5 epoch；GAE 对 terminated/timeout 取 bootstrap=0，仅采样边界用 last_value，done 后不跨。
- **B-V6-6（阻断，不完整）**：没有 entropy 项并入 PPO loss；没有 KL 计算函数；没有“先按完整逻辑 minibatch 算 KL 定 LR，再累计微批梯度、裁剪后一次 Adam”的训练步；没有教师永久 eval/`update_critic_statistics` 禁用/normalizer 冻结的强制。T3 目前是参数与数学库，不是可执行的 S2 PPO 训练循环。

## 4. s2_resources（§2.3）

- REQUIREMENTS 覆盖：GPU ≤7168 MiB、host ≥4096 MiB、5 热身/50 测量/1 恢复、48 步 rollout、20 Adam、独立副本、无 OOM/非有限/接触溢出、评估 100 环境。
- `probe_plan` 计费：`N×(5+50+1)×48 + 100×timeout`，物理子步 4×；`choose_resource` 保留 1024 失败证据后选 512。
- **B-V6-3（阻断）**：`choose_resource([p512])` 可直接选 512，未强制“先 1024、失败才 512”；`train_a2_v09_s2.py --num-envs 512` 也直接接受。用户裁定要求 1024 优先且降档留失败证据。
- 观察：`s2_resources.probe_plan` 未计独立录像（1×timeout），而 `s2_budget` 探测含录像；两处口径需统一；`evaluation_timeout_steps=9472` 为魔法数（预算侧为 9471）。

## 5. s2_checkpoint（§10.1）

- schema 11、stage S2、code/SDS/recipe/map/source/architecture 身份齐全；REQUIRED_METADATA 覆盖课程/强度、模型/优化器/std/normalizer、教师/β、学习/评估/工程/回退消耗、门禁/病例 SHA/候选 SHA/A-B 序列/次数/重跑原因、随机状态/角色/随机化、命令/延迟、入口权重、ledger run、code/manifest。
- **B-V6-5（阻断）**：模块只有 `save_checkpoint`/`validate_metadata`，没有 loader，无法执行“跨版本不加载 6400 旧课程终态”，也无法禁止外部账本回退；`gates_passed` 只查 3 个键（无 A-B 序列）、`final_uses`/`engineering_rerun_reasons`/`teacher_source`/`entry_weight` 未做身份校验；`model_state`/`optimizer_state` 允许 None。

## 6. s2_ledger（Q021）

- 命名空间 `a2-v09-s2-sequential-r2`；header caps；类别 learning/engineering/evaluation；learning 与非学习分列；预留非消费（reserve/commit 分开）；`s1_samples_reused`/`double_counting` 必须 False；非学习 ≤4 亿、学习 ≤36 亿、总 ≤40 亿、S2.0 子额度 ≤0.5 亿；plan SHA/序列/哈希链。
- **B-V6-4（阻断）**：`reserve`/`commit` 不按 `operation_id` 去重——同一 op 连续 reserve 两次被接受（实测 o1 reserved=20）。Q021/SDS 要求逐项去重。
- 观察：无 obligation close；重复 reserve 的余量校验仍按 limit，无法识别同操作重复。

## 7. s2_acceptance（§6.5）

- 27 组×100=2700；每组阈值 95；pitch RMS ≤5、P99 ≤10、>15° ≥0.2 s、速度 RMSE ≤0.10、nonfoot、cruise stall；停车恢复 ≤3 s、末 10 s 平均 ≤0.05、峰值 ≤0.10、位移 ≤0.10、支撑 ≥0.90；工程异常使整套无效。
- **B-V6-2（阻断）**：① 停车缺 `stand_roll_deg/stand_pitch_deg ≤10°` 与 `stand_base_height≥0.30` 检查（实测 30°/30°/0.1 仍返回空原因）；② 不检查 `cruise_window_valid`（无有效巡航窗仍可合格）；③ `final_acceptance` 只校验总 2700，不校验 27 组各 100（50/50 组也能过 95 阈值）；④ 9 个平地组相对 6400 的保持门槛完全缺失；⑤ 平台窗只依赖传入 metrics，未自行验证“四足入台后 1 m”。

## 8. train CLI

- `--dry-run` 默认 1024、`choices` 仅 1024/512；非 dry-run 退出码 1 并提示需 Q021 预留+GPU 窗口；无 `--allow-physics` 旁路；`--max-envs` 已限 1..20。
- 缺口：`--num-envs 512` 直接可用，不需要 1024 失败证据（与 B-V6-3 同源）。

## 9. Q027 / V5 阻断修复

- Q027：`t0=0, t0+1=.04, t0+24=.96, t0+25=1`；失效重入新 `t0`；flat=1；平台首有效样本；快照 `q027_ramp_total_0p5s`、pre-Q027 报错；奖励 pitch/超速同帧同 g。均通过。
- V5 修复复核：H 六例改 `legacy_stairs`（`legacy_h=True`），route 2.0+top+四足不再成功，route 2.6 且 `v07_success` 才成功；六例 timeout 仍 1301/1302/1306/1315/1318/1332。dry-run 输出 `run_plan_batches=30`、`--max-envs 21` 被拒（exit 2）。`s2_state` docstring 已按 Q027 改写（无 provisional/hold）。`case_record` 已加 `config_sha256` 与 `shared_identity_sha256`。

## 10. Q028/Q029（g_level 证据）

**强制向量独立复算**（26 条采样，g=0,1/25,…,1；窗口全含）：

| 量 | 独立期望 | 实测 | 判定 |
|---|---:|---:|---|
| 四足入台/完整几何/g>0/g=1 控制步 | 0/0/1/25 | 0/0/1/25 | 通过 |
| 对应延迟（s） | 0 / 0 / 0.02 / 0.50 | 0 / 0 / 0.02 / 0.50 | 通过 |
| `N` | 26 | 26 | 通过 |
| `Σg` | 13 | 13.0 | 通过 |
| 平均强度 | 0.5 | 0.5 | 通过 |
| 启用占比 | 25/26 | 25/26 | 通过 |
| 最长完全关闭（g==0） | 0.02 s | 0.02 | 通过 |
| 最长未满强度（g<1） | 0.50 s | 0.50 | 通过 |
| N=0 | NA | None/NA | 通过 |

- Q028 分量：`N`/`Σg`/平均/启用占比分字段，无未定义 `duty`；窗口掩码用四足入台后 +1 m（平台起步自出生 +1 m）；末个正巡航步计入、DECEL 前一步为止；`g==0` 与 `g<1` 两个最长连续量分开；只看窗口内、N=0 为 NA；`GeometryFailureTracker` 仅在条件 True→False 记事件（初始不满足不记、重复零帧不重记），原因分类 feet/body/both/other。
- Q029 分量：四个事件的控制步与仿真时刻、相对四足入台延迟分别输出，参考事件自身 0、缺失 NA+原因；中断保留正/满区间与重入。
- **B-V6-7（阻断/缺口）**：
  1. 首米窗口未实现（`first_meter_window=False`），Q028 第 8 条要求同算法单列首米。
  2. `first_full_interval_start` 返回首个 `g>0` 区间起点（1），不是完整几何条件区间起点 `t0=0`，也不是首次 `g=1`（25）；Q029 第 4 条“首次满值标注对应的区间起点”口径不明确/不符。
  3. 旧 `g_level_report` 仍暴露 `duty_cycle`/`max_off_run_steps`，与 Q028 第 5 条“两种占比不得共用无定义 duty”冲突（新函数已分开，但旧 API 未废弃）。
  4. `four_foot_step`/`full_geometry_step` 被当作数组下标使用（`step_values[index]`），若步号非 0 基连续会错位；平台起步以出生四足投影确认入台未在报告中验证。

## 11. 判定与待裁定

- **通过**：rewards 权重/一次性/失败优先/抬脚、loss 精确聚合、ledger 命名空间与 cap、Q027、V5 修复、Q028/Q029 强制向量。
- **阻断**：B-V6-1（stable_stand DECEL）、B-V6-2（acceptance 质量/组成/6400）、B-V6-3（512 先选）、B-V6-4（ledger op 去重）、B-V6-5（checkpoint loader/身份）、B-V6-6（PPO 训练环不完整）、B-V6-7（Q028/Q029 首米与区间口径）。
- **待裁定**：Q-V6-1 acceptance 27×100 与 6400 保持的权威清单；Q-V6-2 runner PPO 完整范围；Q-V6-3 checkpoint loader/教师绑定；Q-V6-4 Q028 首米窗；Q-V6-5 Q029“首次满值区间起点”定义。
- 144 项 S2 CPU 测试通过，但上述阻断项均未被现有测试覆盖（反例已在本报告列出）。

机器可读明细见 `logs/a2_v09/s2_0/verification/train_stack_audit_20261122_r6.json`。
