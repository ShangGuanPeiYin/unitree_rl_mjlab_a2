# A2 v0.9 S2.0 实施缺口、病例冻结与分段指标设计（v2）

**状态：** 纯 CPU 实施记录；未运行物理、未训练、未做 Git 写操作
**Supersedes：** `docs/v09_s2_0_implementation_plan.md`（SHA256 `a441378d401497af070c271efe54ce0634f8b7e08466dc9885b6eb7754bda2be`）
**绑定：** SDS v1.2 `09fa2d4cdd858914c46a51fd52670c20cb20f2b6f3e76e6d266054574f64ccbe`
**S1 历史证据：** 继续绑定 SDS v1.1 `500ad32aaa946c927459427f3a65565b15d4b77fca4be214817121d15c742976`，不改写
**规划裁定：** DSH-Q015～Q025（`doc/sds/reviews/v09_execution_rulings.md`）
**实现子 agent：** A2-S2-IMPL

本文件只描述 S2.0 的输入冻结基础、指标纯函数、诊断消融和预算上界；行为结论必须由带物理的 S2.0 运行产生。

---

## 1. 身份（Q015/Q022）

| 上下文 | 文件 SHA | 使用范围 | 校验入口 |
|---|---|---|---|
| S1（历史） | `500ad32a…` | S1 生成器、流盐、种子、case dict、manifest 的所有参与哈希字段 | `v09/spec.py` 原值未改 |
| S2（实现） | `09fa2d4c…` | 新 S2 原生流、清单、预算、证据 | `s2_spec.validate_sds_context('S2')` |

- `validate_sds_context` 只接受精确 `S1` 或 `S2`；当前 SDS 是 v1.2，`S1` 上下文按设计抛错。
- 正式 S2.0 键：`stage=S2, suite=S2_DIAG`；课程名仍 `S2.0`；`map_branch` 是操作标识，不参与抽样，两支共享同一 case 与流。
- 旧清单 `5efda5ab…` 保留为“零物理、已替代”记录，不删、不覆盖；新清单 `supersedes` 指向它。
- S2 入口：`s1_least.pt`，SHA256 `efa9caa8…`，与 S1.2 原件同字节；S1 主段 100,024,320 样本；原 `gate_A` 未通过记录保留；不生成 `S1_RESTORED`。

---

## 2. 现有能力对照与实施缺口

### 2.1 地形、任务与状态机

| 能力 | 现有实现 | 状态 | 缺口/归属 |
|---|---|---|---|
| S2 地形 0.20/0.20/0.80、平台长 20 m、宽 0.8/4 m、顶高 8 m | 只有 v07 23-tile 表（平台长 2 m） | **缺** | `s2_terrain.py`；`s2_cases.s2_terrain_rows()` 只冻结参数契约 |
| 六任务适配 | `V09S1Env` 只接受 `flat_walk/flat_stop/stairs` | **缺** | `s2_env.py` |
| 状态机 `APPROACH→CLIMB→TOP_TRANSITION→POST_WALK→DECEL→HOLD→SUCCESS` | v07 有 climb/first-step 诊断 | **缺** | `s2_env.py` + `s2_state.py` |
| 单次 `top_arrived`、登顶后不 reset、`D_post` | 无 | **缺** | `s2_env.py`；指标已支持 |
| 距离触发步末、下一 Actor 零、100 步 ramp | S1 只有平地固定索引 | **缺** | `s2_env.py` |
| HOLD 1000 步、末 500 步站立窗 | S1 平地质量窗 | **缺 S2** | `s2_env.py` / `s2_acceptance.py` |
| §4.2 超时公式 | 无 | **已完成** | `s2_spec.timeout_steps` |
| 平地硬 timeout 20.02/28.02/30.02 s | S1 同类值 | **S2 常量已冻结** | `s2_spec.flat_schedule/flat_stop_schedule` |
| `c_request/c_exec/a_request/a_applied` | 部分命令/动作记录 | **缺 S2 全链** | `s2_env.py` |
| `g_level` 0/1、0.5 s 渐进、失效归零 | 无 | **缺** | `s2_env.py` + `s2_gate.py` |
| 抬脚 0.075/0.22、逐足真实表面 | v07 有平地/楼梯分支 | **缺逐足选择** | `s2_rewards.py` |
| 平台真实边界、宽平台不裁窄 | v07 固定 0.8 m | **缺** | `s2_terrain.py` + `s2_env.py` |

### 2.2 奖励、损失与训练

| 能力 | 状态 | 归属 |
|---|---|---|
| §5.1 总公式、§5.3 权重与减速列 | **缺** | `s2_rewards.py` |
| `g_level×pitch`、水平面超速、稳定站立 +1 | **缺** | `s2_rewards.py` |
| 停车关闭前进/腾空并清累计 | **缺** | `s2_rewards.py` |
| §5.4 平地保持 `L_anchor`、`β_flat=0.1`、clean 分母 | **缺** | `s2_loss.py` |
| S2 PPO 新 Adam、std 解冻、KL 调 LR | **缺** | `s2_runner.py` |
| 1024→512 资源准入 | **缺**（顺序已记录 `S2_ENV_ORDER_RULING`） | `s2_resources.py` |
| schema 11 S2 检查点 | **缺**（S1 拒绝 stage≠S1） | `s2_checkpoint.py` |
| S2 独立台账命名空间 | **缺**（S1 header 绑定 v1.1） | `s2_ledger.py` |
| 门禁/终验清单、§6.5 验收 | **缺** | `s2_acceptance.py` / `s2_gate_cases.py` |
| S2 视频/诊断叠加 | **缺** | `s2_diagnostics.py` / `s2_video.py` |

### 2.3 S2.0 诊断与指标

| 能力 | 状态 | 位置 |
|---|---|---|
| 15 组×5 例冻结、两分支、150 条运行计划 | **已完成** | `s2_cases.py` + `logs/a2_v09/s2_0/cases/` |
| 六例 H 外层包装、原 case 不变 | **已完成** | `s2_cases.py` |
| 事件、首米/后段/前足→四足/停车后分段 | **已完成** | `s2_metrics.py` |
| 俯仰正负、角速度、前足支撑/法向力、速度/c_exec | **已完成** | `s2_metrics.py` |
| 步幅/步频/完整周期、缺测、组计数 | **已完成** | `s2_metrics.py` |
| 平台巡航/楼梯/接近段失败分类 | **已完成** | `s2_metrics.failure_context` |
| 原失败位置覆盖判断（未验证标记） | **已完成** | `s2_metrics.original_failure_coverage` |
| Q016 高度通道消融 + Q023/Q024 证据 | **已完成** | `s2_map_ablation.py` |
| Q025 质量窗 | **已完成** | `s2_metrics.quality_window` |
| 真实轨迹采集、contact pair、PD 目标/实际、地图有效性/采样时刻 | **缺** | `s2_env.py` + 证据导出 |
| S2.0 执行 CLI / 报告 | **缺** | `scripts/evaluate_a2_v09_s2_0.py` |

---

## 3. 冻结件与过程纪律

**只追加不覆盖。** 任何 manifest/清单/预算表写下后，修正必须写新版本文件（`*_v2.json` 等），保留旧件并记录 `supersedes` 与旧 SHA。`_write_versioned` 已按此实现；旧 `S2_0.json` 未删、未改。

| 文件 | SHA256 | 说明 |
|---|---|---|
| `logs/a2_v09/s2_0/cases/S2_0.json` | file `09e2fc46…`；manifest `5efda5ab…` | 旧 20 例/组清单，零物理、已替代，保留 |
| `logs/a2_v09/s2_0/cases/S2_DIAG.json` | file `b990052c16a7f3cf1c9f09813805556e1f892120eeb13a3cb7b30cf870153b7a`；manifest `fab7372ec20f44a8f0f8c2a75fe5f179c7beafc390ad22daaf73d8aa42c183a1` | 新 75 例/15 组；`supersedes` 旧 `5efda5ab…` |
| `logs/a2_v09/s2_0/cases/S2_DIAG_run_plan.json` | file `bc9ac8ba0748c9cb9d497a6618e0341fa167f11814990568fc104501d7e8f8c1`；plan `3f09c0091f2ccaab8d87ae362f8fe1595c3e560dce55e4ab02a1fa0f3c98e126` | 150 条分支轨迹 |
| `logs/a2_v09/s2_0/cases/h_series_case0.json` | file `1b0ad3529869a9037f93d3171ac7eeea0965c0401661df34e7c5ffc9afac1ee6`；manifest `22ea4111fddec12dbab1d4d89b26535cc53fdb122dbe661e65ae35b1109f517c` | 六例 H 包装，保留 |
| `logs/a2_v09/s2_0/cases/freeze_receipt_S2_DIAG.json` | 见文件 | 新回执 + 预算草案 |
| `logs/a2_v09/s2_0/cases/history/` | `history_index.json`、`history_index_v2.json` | 旧件副本与覆盖事故记录 |

**历史事故：** H 包装 `32f2ffdd…` 曾在未隔离旧字节时被 `22ea4111…` 覆盖。旧 SHA 已记录在 `history_index.json`，旧字节不可恢复；该记录不美化。

---

## 4. S2.0 病例矩阵（Q020/Q022/Q025）

- 15 组 × 固定 `case_id 0..4`，共 75 个原生 case；每例两地图分支 → 150 条轨迹。
- 平地：3 速 × 20 s；L40（h=0.20、d=0.20、宽 0.80）后走 5 m；8 m 高平台直接起步走 5 m；平台长 20 m；两宽×三速。
- 平台起步：`local_x = 8.0 + 1.0 = 9.0`，z=8.4；y/yaw 用 L 范围；无 `top_arrived`。
- 完整登阶：spawn 用 L 范围；`hard_timeout_steps` 用该例实际 approach 的 §4.2 公式。
- Q025：平台质量窗 `[spawn+1 m, spawn+5 m]`（名义评价段 4 m）；首米单列；超时仍 5 m，不额外加 1 m。完整登阶质量窗自四足入台 +1 m 起；`D_post` 自 `top_arrived` 起，两者不混算。
- 两分支共用 case/流/权重/代码；`map_branch` 不参与抽样。

---

## 5. 六个 H 病例（Q017/Q018）

外层 suite `S2_DIAG`；内层 case 逐字段保留原 `S1/S1_DIAG` 内容、原 `case_sha256`、stage/suite 不改；外层记录 source/path/原 case_sha/独立 `operation_id`。

| lesson | h (m) | 原 v07 timeout | 终止机身 x | 终止段 | 距平台前缘 | 终止 platform_support |
|---|---:|---:|---:|---|---:|---|
| H00 | 0.020000 | 1301 | 2.278524 | platform | +0.478524 | [F,F,F,F] |
| H01 | 0.030833 | 1302 | 2.571113 | platform | +0.771113 | [F,T,F,F] |
| H03 | 0.052500 | 1306 | 2.657047 | platform | +0.857047 | [F,F,F,F] |
| H06 | 0.085000 | 1315 | 2.536059 | platform | +0.736059 | [F,F,F,F] |
| H07 | 0.095833 | 1318 | 1.033322 | stairs | −0.766678 | [F,F,F,F] |
| H10 | 0.128333 | 1332 | 1.203684 | stairs | −0.596316 | [F,F,F,F] |

- 终止位置取 `episode.forward_distance_end`；`forward_distance_terminal` 为减 spawn 的带符号距离；`route_final` 不是终止机身位置。
- H00/H01/H03/H06 机身终止于平台，但 H00/H03/H06 四足均无支撑、H01 仅右前足；不得统一写成“平台巡航摔倒”。H07/H10 终止于楼梯。
- 超时：原 v07 逐例；实际和 7,874，同批 padding 上界 7,992，预留 9,432。
- 缺字段：原始 qpos/qvel、原始 RNG、全速率帧审计；**另加“首足/四足入台归因”缺字段补采项（≤6 条，计入 S2.0 子额度）**，因原 JSON 无逐帧足位/接触。
- `original_failure_coverage` 用 `max_route_x` 对比原终止 x（默认 0.05 m），标 `covered / not_covered` 与 `surface_match`；未到达原位置标“未验证”。

---

## 6. 分段指标（Q019/Q025）

### 6.1 事件

- `front_foot_entry`（别名 `front_foot_top_support`）：首个前足同时满足 §5.3 顶面支撑且 XY 在平台内。
- `four_feet_platform`：四足 XY 首次全部在平台内。
- `g_level_positive/full`、`top_arrived`、`failure`、`stop_request`、`terminal` 首帧锁存。
- 平台矩形 `[front_x, front_x+length] × |y|≤width/2`，边界 `1e-9`。

### 6.2 分段

| 段 | 起点 | 终点 |
|---|---|---|
| `front_entry_to_four_entry` | 前足入台 | 四足入台或失败/终止 |
| `first_meter` | 四足入台（平台为出生） | 净位移 ≥1.0 m 或失败/停车/终止 |
| `post_first_meter` | 首米末点 | 停车/失败/终止 |
| `after_stop_request` | 停车请求 | 终止 |
| `quality_window`（Q025） | 零点 +1 m | 平台为 +5 m；完整登阶到减速/终止 |

失败同帧优先：目标达成与 `failure` 同帧时 `completed=False, failed=True, end_reason='failure'`。某段起点晚于失败帧时 `entered=False, not_reached=True`。

### 6.3 指标

- 净位移 `route_x[end]-route_x[start]`；`D_post` 另用 `top_arrived`/平台出生零点。
- 俯仰 `max/min` 有符号（deg）、`abs_max`、`sign_convention`；角速度单样本段给 `None`。
- 前足 `any/all` 支撑比例；法向力 mean/min/max。
- 实际路线速度、`c_exec` mean/min/max。
- 每足完整落地周期；<2 周期缺测，`None` + 理由；组内只对有效步态取中位数。
- 组计数 `entered/failed/not_reached/completed/valid_gait_samples`，失败段 `platform/stair/approach`。
- `quality_window` 含 `start_m/end_m/nominal_length_m/metrics`。

---

## 7. 高度通道消融与证据（Q016/Q023/Q024）

- 只替换有效格高度；valid/age/坐标/10 Hz/采样帧不变；无效格 `0/0/1`。
- 公式：`clip((z_plane_world - z_base_sample)/1.5, -1, 1)`。
- 完整上楼：前足首次真实顶面支撑事件帧 `t` **必用原图**；`s = 严格大于 t 的首个正常新采样` 才首次消费替换图；不补刷、不回写、不跳过 `t+1` 新采样。`t` 后即终止记“事件已触发但未暴露”。平地/平台自出生首个采样生效。
- Q023 分母＝Actor 实际消费帧数（回合首帧到首次终止前最后一次 Actor 调用，inclusive；不算 reset 后帧与 padding；同图连续复用逐帧计）：
  - `valid_fraction = Σ有效格帧数 /(帧数×40×30)`；
  - `saturation = Σ(valid 且 |h输入|≥0.999)/Σvalid`，窗口＝首次替换被 Actor 看见起至结束；
  - `exact_clip_fraction = Σ(valid 且 |未裁剪归一化高度|≥1)/同一分母`；
  - 存原始分子/分母；组汇总分子分母相加；未暴露、零有效格单列；零分母记 `NA`。
- Q024 证据：共用 case/流/权重/代码；配置 diff 仅诊断开关/记录标识；干预前配对状态/输入/动作摘要；同状态同采样帧原图 vs 替换图对照（只允许有效高度改变）；`paired_proof` 校验 valid/age 不变、仅有效格高度变。
- 只用于 15 组 S2.0；H/训练/门禁/终验真实图。

---

## 8. 预算（Q021，纯 CPU 上界）

### 8.1 S2.0（新清单 75 例）

| 项目 | 控制样本 |
|---|---:|
| 真实图：15 批×5 例（真实 spawn/实际 timeout + padding） | 210,195 |
| 平面图：同 15 批 | 210,195 |
| 两分支合计（150 条） | 420,390 |
| 六例 H 原 timeout 实际和 | 7,874 |
| 六例同批 padding 上界（6×1332） | 7,992 |
| 六例预留 | 9,432 |
| **S2.0 上界合计** | **429,822** |
| 0.5 亿剩余 | 49,570,178 |

旧 `429,722` 是旧 `S2_0` 清单上界，不再使用；旧 `467,830` 不使用。

### 8.2 门禁（≤40 次，取最坏课程）

| 课程 | 组数 | 例数 | 20 例/组上界 |
|---|---:|---:|---:|
| S2.1 | 36 | 720 | 1,620,000 |
| S2.2 | 33 | 660 | 2,061,980 |
| S2.3 | 51 | 1020 | 4,348,500 |
| 40 次（取 S2.3） | — | — | **173,940,000** |

### 8.3 终验（§6.5，27 组×100 例）

| 项目 | 控制样本 |
|---|---:|
| 候选 2,700 例 | 13,249,500 |
| S1 冻结对照 2,700 例 | 13,249,500 |
| 原 6400 平地 900 例 | 1,170,900 |
| 每套 | 27,669,900 |
| 三套 | **83,009,700** |

### 8.4 资源探测（§2.3 全套，1024 优先→512）

| 项目 | 1024 | 512 |
|---|---:|---:|
| 5 热身 + 50 测量（55×N×48） | 2,703,360 | 1,351,680 |
| 100 环境评估（9,472 步） | 947,200 | 947,200 |
| 1 环境视频 | 9,472 | 9,472 |
| 恢复 rollout | 49,152 | 24,576 |
| **合计** | **3,709,184** | **2,332,928** |

### 8.5 全阶段原子预留（Q021）

| 项目 | 控制样本 |
|---|---:|
| 资源探测 1024 | 3,709,184 |
| 资源探测 512（回退预留） | 2,332,928 |
| S2.0 两分支（新清单） | 420,390 |
| 六例 H 缺字段补采（≤6 条） | 9,432 |
| 六例候选复测（3 候选×9,432） | 28,296 |
| §6.1.1 旧 L40 里程碑一次性冻结对照缓存 | 438,440 |
| ≤40 次门禁 | 173,940,000 |
| 三套终验 + 冻结对照 | 83,009,700 |
| **合计** | **263,888,370** |
| 4 亿剩余 | 136,111,630 |

**单次计费/探测口径：** 单位=环境控制样本；物理子步 4×另记不重复收费；每批按 `实际批环境数×该批最长硬 timeout`（含完成 slot 等待空步、padding）；reset 后帧不计；S1 已计费样本不计入 S2；探测含整卡峰值 ≤7168 MiB、主机可用 ≥4 GiB、5 热身 + 50 测量 + 一次恢复检查（完整 48 步 rollout + 5×4=20 次 Adam）、独立副本、100 环境评估。

**留出假设：** 40 次门禁按阶段总次数（SDS `GATE_LIMITS['S2']=40`）每次取最坏 S2.3；若裁定为每课程 40 次则超 4 亿。候选复测按 3 候选各一次六例。门禁/终验清单未冻结，正式 Q021 预留前必须用真实 manifest/候选数/padding 重算。

### 8.6 学习额度

| 课程 | 学习上限 | 1024 rollout | 512 rollout |
|---|---:|---:|---:|
| S2.1 | 8 亿 | 16,276 | 32,552 |
| S2.2 | 12 亿 | 24,414 | 48,828 |
| S2.3 | 12 亿 | 24,414 | 48,828 |
| 合计 | 32 亿 | 65,104 | 130,208 |

每轮样本：1024×48=49,152；512×48=24,576。S2 学习上限 36 亿，余 4 亿未分配。

---

## 9. 测试

```bash
/home/xxs/anaconda3/envs/a2/bin/python -m pytest -q \
  tests/test_v09_s2_0_cases.py tests/test_v09_s2_0_metrics.py \
  tests/test_v09_s2_0_map_ablation.py tests/test_v09_s2_0_budget.py
```

**结果：64 passed。**

| 文件 | 数量 | 覆盖 |
|---|---:|---|
| `test_v09_s2_0_cases.py` | 15 | 双 SHA、S2_DIAG/supersedes、旧件保留、150 计划、流独立、5 例、平台 9.0/8.4、L40、超时、Q025 常量、H 包装/原 SHA/终止位置/内层不变、地形、S1 未动 |
| `test_v09_s2_0_metrics.py` | 26 | 事件、1 cm 边界、1.0/0.999、同帧失败、早失败、缺测、单样本、正负俯仰、g_level 延迟、停车、零点、负位移、组计数、平地对照、D_post、失败分类、覆盖、足迹区分、Q025 质量窗 |
| `test_v09_s2_0_map_ablation.py` | 13 | 仅高度、0/0/1、valid/age、mask、Q023 分母/饱和/精确裁剪/NA、Q024 同帧/保持帧/终止未暴露/出生、缓存复用、组汇总、配对证明 |
| `test_v09_s2_0_budget.py` | 10 | 新/旧清单、5 例分批、H 7874/7992/9432、门禁 36/33/51、终验 27/3 套、L40 缓存 438,440、probe、总上界、billing、Q025 敏感性 |

已有 v09 测试（排除 `test_v09_resource_cli.py`）：259 passed；该文件 21 项 legacy CLI 失败与本任务新增文件无关，未宣称通过。

---

## 10. 裁定落实

| 编号 | 落实 |
|---|---|
| Q015 | 双上下文校验；S1 常量/流盐/case 未改 |
| Q016 | 仅有效格高度；clip 公式；分支/latch |
| Q017 | H 外层 S2_DIAG，source/path/case_sha/operation_id；内层原样 |
| Q018 | 原 v07 1301/1302/1306/1315/1318/1332；7,874/7,992/9,432；失败段按终止位置/足迹区分 |
| Q019 | D_post 零点、首米零点、平台出生、带符号净位移 |
| Q020 | 15×5；平地 20 s、L40+5 m、8 m 平台+5 m；两分支=150 |
| Q021 | 全阶段上界 263,888,370 ≤4 亿；含 L40 缓存/等待空步/去重；未写台账 |
| Q022 | 正式 suite S2_DIAG；旧 5efda5ab 保留；新清单 supersedes |
| Q023 | 固定分母、饱和/精确裁剪、原始分子分母、NA、组汇总 |
| Q024 | 触发帧原图、严格下一新采样替换、未暴露、配对/config 证据 |
| Q025 | 平台窗 +1～+5 m；首米单列；超时不加步；敏感性 334/200/167 |

**未决语义项：无。** 剩余为实现缺口（§2），不是待裁定语义。

---

## 11. 下一步

1. `s2_terrain.py`：编译 40 级/20 m 平台/两宽真实地形与解析表面；碰撞-地图同源测试。
2. `s2_env.py` + `s2_state.py`：六任务、状态机、`g_level`、`D_post`、停车/保持、超时；接入指标/消融。
3. `s2_rewards.py`/`s2_loss.py`：§5、§5.4。
4. `s2_checkpoint.py`/`s2_ledger.py`：schema 11、独立命名空间、门禁/终验计数。
5. `s2_acceptance.py`：§6.5 27 组/6300 条预算与质量。
6. `scripts/evaluate_a2_v09_s2_0.py`：仅在主对话登记预算与执行窗口后运行。

---

## 12. 未执行项

- 未训练、未仿真、未 `env.step`/mjwarp、未评估 rollout、未用 GPU、未梯度更新。
- 未做任何 Git 写操作；未修改 SDS、`logs/codex_changes.md`、`scripts/play.py`、`scripts/train.py` 或既有 S1 v09 文件。
- 未生成 `S1_RESTORED`；原 `gate_A` 未通过记录保持。
