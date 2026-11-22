# A2 v0.9 S2.0-V3 裁定、独立重生成与预算核对报告

- 审核 agent：A2-S2-VERIFY（独立验收子 agent，未参与实现）
- 审核时间：2026-11-22（快照 14:15–14:24）
- 仓库/分支/HEAD：`/home/xxs/dog/unitree_rl_mjlab_a2` / `main` / `52557bd491f37c5e724ae92126cc8b47e7ea7e38`
- 规范：SDS v1.2（`09fa2d4c…`）；裁定：`doc/sds/reviews/v09_execution_rulings.md`（Q022～Q025）
- 机器可读件：`logs/a2_v09/s2_0/verification/rulings_audit_20261122_r3.json`
- 独立复算：`logs/a2_v09/s2_0/verification/recompute_s2_diag_v03.py`（生成 `s2_diag_expected_20261122.json`）
- 反例测试：`logs/a2_v09/s2_0/verification/test_s2_ablation_counterexamples_v03.py`
- 范围：纯 CPU；未做 GPU/物理/训练、未改实现/测试/SDS/台账/checkpoint、未做 git 写。

## 0. 结论分级

| 项 | 结论 | 关键事实 |
|---|---|---|
| Q022～Q025 留档 | **通过** | `3b36083…`＝commit `52557bd` 内容；4 节齐全；本地＝origin/main |
| 新 `S2_DIAG` 清单（75 例） | **通过** | 独立重生成逐字段 **0 处不符**（含 `case_sha`、流、初态、几何、时限） |
| 150 条分支运行计划 | **通过** | 150 entries＝75 例×2 分支；30 批；`plan_sha 3f09c009…` |
| 旧 `5efda5ab…` 处理 | **通过（注记位置待补）** | 旧件仍在；新 receipt `supersedes` 注明"零物理、已替代"；旧件/history index 本身无标记 |
| 新预算 | **通过** | 单分支 210,195；两分支 420,390；含 H 9,432 → **429,822**；`429,722` 作废 |
| Q023 字段实现 | **阻断（1 项）+ 待补** | 未暴露终止时分母越界；真实支/逐步比对/报告集成缺失 |
| Q024 同帧语义 | **阻断（同一项）** | t 用原图、s>t、不补刷均正确；未暴露终止守卫失败 |
| Q021 预留完整性 | **通过（清单）；正式预留未写** | 263,888,370 ≤ 4 亿；新增 L40 缓存 438,440；`q021_reservation_written=False` |
| **总判定** | **S2.0 输入/病例/预算通过；Q023/Q024 证据模块 1 项阻断** | 见 B-V03-1 |

## 1. 留档核对（Q022～Q025）

- 文件 SHA `3b36083224d5c8e334ddcb9fc60449fa9ad340ac5890f79e5bae0c1221b4012f`，与 `git show 52557bd:doc/sds/reviews/v09_execution_rulings.md` 完全相同，工作区无改动。
- commit `52557bd491f37c5e724ae92126cc8b47e7ea7e38`（`docs: record S2 rulings DSH-Q022..Q025`，14:15:12）＝`git ls-remote origin refs/heads/main`。
- 4 节齐全、接续 Q021：Q022 命名与 suite、Q023 证据字段、Q024 同帧语义、Q025 平台 5 m 含首米。
- 与转述/SDS 逐项核对：算术与数值一致；未发现改写走样。限制同前：规划原文不在工作区，未逐字比对。
- 文档抬头仍写"绑定 SDS v1.1"，Q015～Q025 实际按 v1.2，建议注明抬头仅适用 Q001～Q010（沿用 V2 观察）。

## 2. 独立重生成（Q022）

### 2.1 公式与验证方法

- 生成器：`GEN='v09-s2-sha256-u53-v1'`，盐 `SDS v1.2 09fa2d4c…`，seed 42，`stage=S2`，`suite=S2_DIAG`。
- 组角色：`role = sha256({stage,task,stair_count,geometry,speed,stop_mode,force_family})`。
- 流：`stream = sha256([GEN, SDSv1.2, 42, 'S2', 'S2_DIAG', role, case_id, channel])`，`channel∈{spawn_x,spawn_y,spawn_yaw}`。
- 抽样：u53 `fraction=(first16>>11)/2^53`，`x∈U[-1.1,-0.9]`、`y∈U[-0.05,0.05]`、`yaw∈U[±5°]`；平台直接起步 `x=平台前缘+1.0`（Q019）。
- **先自证公式**：用同一实现无关脚本以 `suite=S2_0` 重建旧清单，**75/75 逐字节一致**（含 `case_sha`），再改为 `S2_DIAG` 生成新 75 例。

### 2.2 与实现方冻结件比对

冻结件：`logs/a2_v09/s2_0/cases/S2_DIAG.json`（file `b990052c…`，manifest `fab7372e…`，`validate`=True；150 条运行计划 `validate`=True）。

- 75 例×14 个字段（suite/stage/streams/spawn/hard_timeout/geometry/command/task/speed/stair_count/stop_mode/force_family/case_key/case_id）与我的独立结果 **0 处不符**；`case_sha256` 全部相同。
- 分组最大硬时限：平地 1001；L40 5 m 0.3/0.5/0.6 = 6694/4219/3598（宽 0.8）、6700/4220/3603（宽 4.0）；8 m 平台 5 m = 2167/1500/1334（两宽相同）。
- 单分支 210,195；两分支 420,390；150 条原生轨迹。
- 运行计划：150 entries、`batch_size=5`、`batches=30`、75 例均含 `real` 与 `plane_diagnostic` 两分支；两分支共享同一 case/流；`map_branch` 仅操作标识（符合 Q022 第 2 条）。
- 旧件：`S2_0.json`（`09e2fc46…`/`5efda5ab…`）仍存在；新 receipt `supersedes.reason` = “Q022: formal suite is S2_DIAG; old S2_0 manifest kept as zero-physics superseded record”。但**旧件本体与 `history/history_index.json` 没有 superseded 标记**（index 仍写 `suite: S2_0`），注记只在新的 superseding receipt 中——见 O-V03-1。

## 3. 独立重算预算（Q025）

Q025 确认平台 5 m **含首米**（`x_spawn+1` 开窗、`x_spawn+5` 终点），并明确超时公式**不加** 1 m/额外步数；完整登阶 `D_post=5` 仍自 `top_arrived` 起算。故平台硬时限与 V2 相同，仅 suite 改名导致流/初态变化、从而改变 L40 组的 `L_approach`。

| 组 | 任务 | 速 | 台宽 | 例数 | 批 env | 硬时限 | 批最坏控制步 |
|---|---|---:|---:|---:|---:|---:|---:|
| G01–G03 | flat_walk | 0.3/0.5/0.6 | — | 5 | 5 | 1001 | 5,005×3 |
| G04 | stairs_walk | 0.3 | 0.8 | 5 | 5 | 6694 | 33,470 |
| G05 | platform_walk | 0.3 | 0.8 | 5 | 5 | 2167 | 10,835 |
| G06 | stairs_walk | 0.5 | 0.8 | 5 | 5 | 4219 | 21,095 |
| G07 | platform_walk | 0.5 | 0.8 | 5 | 5 | 1500 | 7,500 |
| G08 | stairs_walk | 0.6 | 0.8 | 5 | 5 | 3598 | 17,990 |
| G09 | platform_walk | 0.6 | 0.8 | 5 | 5 | 1334 | 6,670 |
| G10 | stairs_walk | 0.3 | 4.0 | 5 | 5 | 6700 | 33,500 |
| G11 | platform_walk | 0.3 | 4.0 | 5 | 5 | 2167 | 10,835 |
| G12 | stairs_walk | 0.5 | 4.0 | 5 | 5 | 4220 | 21,100 |
| G13 | platform_walk | 0.5 | 4.0 | 5 | 5 | 1500 | 7,500 |
| G14 | stairs_walk | 0.6 | 4.0 | 5 | 5 | 3603 | 18,015 |
| G15 | platform_walk | 0.6 | 4.0 | 5 | 5 | 1334 | 6,670 |

- 单分支 **210,195**；两地图分支 **420,390** = **150 条原生轨迹**（物理子步 1,681,560）。
- H 六例另计：实跑 7,992 / 预留 **9,432**。
- **含 H 总上界 429,822 控制步 / 1,719,288 物理子步 = 0.8596% × 0.5 亿**。
- 与实现方 receipt 完全一致（420,390 + 9,432 = 429,822）；**`429,722` 已作废**（旧 `S2_0` 清单上界，现仅作历史记录）。
- 敏感性：若误把 Q025 的首米再加 1 m，平台组将各增 ~333/200/167 步（0.3/0.5/0.6），属误读；Q025 明文禁止。

## 4. Q023 字段实现与反例

实现位置：`s2_map_ablation.py`（快照 `2090a910…`）。逐项：

| Q023 要求 | 实现 | 判定 |
|---|---|---|
| 分母＝实际 Actor 消费帧（含首帧至终止前最后一次，含复用帧；不含 reset/padding） | `frames_consumed` 仅在 `consumed=True` 时递增；`consumed=False`/`finished` 后返回 `consumed=False` | 正常路径通过；**未暴露终止时不置 finished**（B-V03-1） |
| `valid_fraction = Σ有效格帧 /(帧×40×30)` | `summary.valid_fraction = valid_cell_frame_sum/(frames_consumed×1200)` | 通过（1080/1200=0.9 实测） |
| `saturation = Σ(valid 且 |h输入|≥0.999)/Σvalid`，自首次替换被看见至结束 | 逐消费帧累计，阈值 0.999，分母为替换窗口内 Σvalid | 通过（远平面 raw=1.067 → 1.0；近平面 → 0） |
| `exact_clip_fraction = Σ(valid 且 |未裁剪|≥1)/同一分母` | `exact_clip_numerator/saturation_denominator` | 通过 |
| 保存原始分子/分母；组汇总分子分母相加；未暴露/零有效格计数；无分母 `NA` | `summary` 保存分子分母；`aggregate_evidence` 分子分母相加并统计 not_exposed/zero_*；None=NA | 通过 |
| 真实支按其自身事件/刷新窗口统计 | **未实现** | 待补（O-V03-5） |
| 逐步检查 valid/age 与同帧未替换输入相同、无效格 0/0/1 | 仅首次替换帧调用一次 `assert_only_height_changed`；`pairing_records` 只存调用方数据 | 不足以满足"逐步"（O-V03-5） |

反例实测（`test_s2_ablation_counterexamples_v03.py`）：

| # | 场景 | 结果 | 判定 |
|---|---|---|---|
| 1 | 出生即暴露（flat） | active、first_consumed=0、valid 1080/1200、sat 分母 1080 | 通过 |
| 2 | 无效格 | 高度 0、valid 0、age 1 | 通过 |
| 3 | 远平面（raw 1.067） | saturation 1.0、exact 1.0 | 通过 |
| 4 | 近平面（raw −0.267） | saturation 0、exact 0 | 通过 |
| 5 | 同帧触发 t（t 有刷新） | t 用原图、active=False；t+1 无刷新仍 False；s=t+2，s−t=2 | 通过（Q024） |
| 6 | 保持帧触发（t 无刷新） | t、t+1 原图；s=t+2，s−t=2 | 通过 |
| 7 | 触发后终止前未暴露 | `finished=False`；终止后仍 `consumed=True`；frames=3（应为 2）；note 正确 | **阻断 B-V03-1** |
| 8 | padding/reset | `consumed=False` 不计数；active 终止后 finished=True | 通过 |
| 9 | 零有效格 | valid_fraction=0、saturation/exact=None(NA) | 通过 |
| 10 | 组汇总 | 分子分母相加、not_exposed/zero 计数正确 | 通过 |

另：`summary()` 在快照 `2090a910…` 已可正常调用（此前 `self.exposed` 崩溃已修复）；证据尚未接入 `s2_metrics`/病例报告（无 `valid_fraction`/`saturation` 字段），见 O-V03-4。

## 5. Q024 同帧语义

- 第 t 步事件：`trigger_frame=t`，`trigger_frame_uses_real=True`，第 t 步返回原图（实测通过）。
- `s = 严格大于 t 的首个 refreshed 步`；同帧刷新不计（`frame_index > trigger_frame`）；t 的旧刷新不回写、不补刷（代码无回写路径）——通过（反例 5/6）。
- 记录 `trigger_frame`、`first_consumed_replacement_frame`、`delay_steps=s−t`；`first_replacement_frame` 只记首次（已修复 V2 的覆盖问题）。
- 缺：**"原生采样帧标识"** 未定义/未保存（只有 Actor 步号）；Q024 最小证据里的 `sample_frame` 仅在调用方配对字段中声明，未被校验。
- 四类覆盖：同帧/保持帧/出生即暴露通过；**终止前未暴露**因 B-V03-1 的 finished 守卫失败而不完全正确（记录本身正确）。

## 6. Q021 预留完整性（新清单）

实现草案（`s2_budget.s2_stage_reservation_draft`，CPU 只读运行，新 `S2_DIAG` 清单）：

| 项 | 控制步 |
|---|---:|
| 1024 探测 | 3,709,184 |
| 512 探测 | 2,332,928 |
| S2.0 两分支 | 420,390 |
| H 六例入口补采 | 9,432 |
| H 候选复测（3 模型） | 28,296 |
| L40 里程碑冻结对照缓存（一次性） | 438,440 |
| 40 次门禁（全按最坏 S2.3） | 173,940,000 |
| 三套终验+冻结对照 | 83,009,700 |
| **合计** | **263,888,370 / 4 亿；余 136,111,630** |
| S2.0 子额度 | 429,822 / 0.5 亿；余 49,570,178 |
| 学习 | 32 亿 / 36 亿（余 4 亿=两次 2 亿恢复） |

- 清单完整性：1024/512 探测、S2.0、H ≤6、候选复测、≤40 门禁、三套终验、冻结对照（含 L40 缓存）、padding/完成 slot 等待、去重、S1 不复用——均有对应项/`billing` 字段；总上界 ≤4 亿。
- 仍待：`q021_reservation_written=False`，尚无 S2 台账原子预留事件；按 Q021 第 1 条，物理仿真前必须完成，故 S2 物理仍不得启动。
- L40 缓存按 A/B 两套计价 438,440（3 组三速×20 例×2 套 A/B，逐速时限合计 10,961 步/例），与"一次缓存"的措辞存在保守偏差（O-V03-7），不影响上界。

## 7. 阻断、观察与待裁定

**阻断**

- **B-V03-1**（Q023/§Q024）：未暴露即终止时 `finished` 未置位，终止后的 `consumed=True` 仍计入 `frames_consumed`（反例：触发 t=30、终止 t=31、额外调用 t=32 → frames=3，应为 2）。违反 Q023 第 1 条"不含 reset 后帧或 padding"，并使四类覆盖中的"终止前未暴露"不成立。修法：在未激活/触发帧的早返回里同样处理 `terminal`，并加回归测试。

**观察**

- O-V03-1 旧 `5efda5ab…` 仍在，新 receipt `supersedes` 已注明"零物理、已替代"；旧文件与 history index 无该标记，index 仍 `suite=S2_0`。
- O-V03-2 独立重生成 75/75 精确一致；先以 S2_0 复现旧清单自证公式，再切 S2_DIAG。
- O-V03-3 新预算 429,822；旧 429,722 作废。
- O-V03-4 Q023 字段只在 `s2_map_ablation`，未接入 `s2_metrics`/病例记录。
- O-V03-5 Q023 第 6 条真实支窗口、第 7 条逐步 valid/age 相等均未实现（仅首替换帧一次 `assert_only_height_changed`）。
- O-V03-6 Q024 原生采样帧标识缺失；最小证据字段声明但未校验；物理/奖励/终止配置身份未产出。
- O-V03-7 L40 缓存按 2 套 A/B 计价；Q021 原文"一次缓存"，属保守上界。
- O-V03-8 `validate_s2_0_manifest` 名称未随 Q022 改名（功能正常）。

**待裁定**

- Q-V03-1｜Q022：旧件"零物理、已替代"标记是否需写入旧文件/history index（当前只在 superseding receipt）。建议补 index。
- Q-V03-2｜Q023：saturation 对缓存地图按每个 Actor 消费帧计（dwell-weighted）是否为裁定本意；建议确认。
- Q-V03-3｜Q024：`原生采样帧标识` 指地图刷新计数、Actor 步或两者；建议两者都记。
- Q-V03-4｜Q023：真实支对应窗口统计的归属（模块 API 还是 metrics 集成）。
- Q-V03-5｜Q021/§6.1.1：L40 缓存按 1 套还是 2 套 A/B 计价（建议保留保守 2 套并注明）。

## 8. 未执行与复现

- 未做 GPU/物理推进、未训练、未补采、未修改实现/测试/SDS/台账/checkpoint、未 git 写。
- 独立重生成：`/home/xxs/anaconda3/envs/a2/bin/python` 运行 `recompute_s2_diag_v03.py`（先断言 75/75 复现 S2_0，再生成 S2_DIAG）。
- 反例：`test_s2_ablation_counterexamples_v03.py`。
- 冻结件校验：`validate_s2_0_manifest(S2_DIAG.json)`、`validate_s2_diag_run_plan`、`validate_h_series_manifest` 均为 True（快照 14:24）。

机器可读明细见 `logs/a2_v09/s2_0/verification/rulings_audit_20261122_r3.json`。
