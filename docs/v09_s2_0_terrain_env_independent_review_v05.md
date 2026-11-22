# A2 v0.9 S2.0-V5 T2 独立验收报告（地形/状态机/env/CLI/g_level）

- 审核 agent：A2-S2-VERIFY（独立验收子 agent，未参与实现）
- 审核时间：2026-11-22（快照 14:34–14:44）
- 仓库/分支/HEAD：`/home/xxs/dog/unitree_rl_mjlab_a2` / `main` / `bd1bc33`
- 范围：T2 交付的 `s2_terrain.py`、`s2_state.py`、`s2_env.py`、`scripts/evaluate_a2_v09_s2_0.py`；含 Q027 `g_level` 复核。纯 CPU，未跑物理/训练。
- 机器可读件：`logs/a2_v09/s2_0/verification/terrain_env_audit_20261122_r5.json`
- 复测脚本：`test_s2_v5_state_gate_rewards.py`、`dry_run_r5*`（均在 verification/ 下）
- 代码快照：terrain `74bc1492…`、state `bf436ce4…`、env `5117f3ac…`、metrics `2cb73170…`、rewards `5a1c6d04…`、CLI `fdf0ef6f…`。

## 0. 逐项结论

| # | 项 | 结论 | 要点 |
|---|---|---|---|
| 1 | 地形/表面查询 | **通过** | 几何/宽台/拼接/valid/MJCF 全部独立复算一致 |
| 2 | 状态机与事件 | **通过（S2.0）；H 复测阻断** | 顺序、单次 top、失败优先、步末距离、超时、双零点均对；H 适配器终止条件与 v07 不同 |
| 3 | 证据契约 | **通过（schema）；配置 SHA 绑定待补** | metrics 所需字段全在 trace；H 足位/接触可导出；config SHA 只在 pairing，不在 case_record |
| 4 | 分支配对 | **通过** | 75 对/150 条；共用流/命令；前缀证明可复算；map_branch 不参与抽样 |
| 5 | CLI dry-run | **阻断** | 30 批未输出；`--max-envs 200` 未拒绝；非 dry-run 正确拒绝 |
| 6 | `g_level`（Q027） | **行为通过；声明阻断** | 离散序列/失效重入/快照版本正确；模块 docstring 仍是“hold 后 ramp / provisional” |
| **总** | | **3 项阻断（H 终止、CLI 契约、g_level 注释）** | 其余通过 |

## 1. 地形与表面查询（通过）

- `terrain_specs()` 7 行：flat + N=40/10/20 × 宽 0.8/4。40 级原点 12/28、10 级 44/60、20 级 76/92；`h=d=0.20`、梯宽 0.80、平台长 20、顶高 N·h、前缘 N·d。
- 逐盒对照：N40 宽 0.8：ground x[-2,28]、y±2、top 0；40 级踏面 `[ (i-1)0.2, i0.2 ]`、y±0.4、top i·0.2；平台 x[8,28]、y±0.4、top 8。宽 4 时平台 y±2、**未裁成 0.8**。
- 拼接：x=8.0 处平台与最后一级踏面同顶 8.0，仅共边界、无重叠面；x=7.99、y=0.41 → ground 0；x=8.01、y=0.9（窄台）→ ground 0、（宽台）→ platform 8.0；无虚假支撑。
- `query_surface`：valid bounds 内取最高盒顶，越界 `nan`/valid=False（与 v07 原约定一致）；`surface_class` 的 ground/tread/platform/outside 与查询一致。
- H00 legacy：N=6、h=.02、d=.3、宽.8、台长 2.0；ground x[-2,3.8]、tread6 top .12、platform x[1.8,3.8] y±.4；查询与分类一致。
- MJCF：153 geoms，`validate_compiled_model` 通过；friction `(0.6,0.005,0.0001)`、condim 3、contype/conaffinity 1，与 v07 地形一致；几何与解析查询同用 `boxes()`。

## 2. 状态机与事件（S2.0 通过；H 阻断）

- 顺序：APPROACH→CLIMB（first_step）→TOP_TRANSITION（top_arrived）→POST_WALK（four feet）→SUCCESS；实测 route 8.0 top、8.1 四足、13.0 达 5 m 成功。
- `top_arrived` 仅锁存一次（重复信号不重复事件），之后不 reset；同帧 `top+failure` → FAILURE 且 top 不锁存；timeout 按 `s2_spec`/冻结值在配置步触发（实测 3 步配置第 3 步失败）。
- 距离在步末触发：成功帧即跨 5 m 的帧，命令轨迹下一步才变化；`D_post` 完整登阶零点=锁存 top 的 `route_x`（8.0），平台起步零点=出生 x（9.0），平台 5 m 含首米（13.0 处 4.0）。
- 平地初始 POST_WALK，1000 步成功；硬 timeout 1001 未被误触。
- **B-V5-2（阻断，H 复测）**：`S2TaskConfig.from_case()` 把 `task='stairs'`（H 六例）映射为 `stairs_walk` 且 `post_distance=0`；S2 机器在 top+四足后即成功。v07 原终止是 body_x≥前缘+0.8 m 的 top 事件+支撑，不是四足全入台。反例：H00 route_x=2.0（前缘 1.8，机身仅过 0.2 m）+top+四足 → S2 机器 SUCCESS；v07 需 body≥2.6。H 复测不得走该成功路径。
- 观察：`step()` 返回全部历史事件（failure 分支只返回新事件），调用方需自行 diff；S2.0 walk 不受影响。

## 3. 证据契约（schema 通过；配置 SHA 待补）

- `TRACE_FIELDS` 27 项是 `s2_metrics` 所需字段（step/route_x/base_xy/feet_xy/pitch/foot_on_platform/foot_contact/g_level 等）的超集；逐帧足位、接触、法向力、contact pairs、PD 目标/实际关节、map valid/sample time、native_sample_id 均在。
- H 六例缺的逐帧足位/接触可由该 recorder 导出（stage=S1 走 legacy 地形，字段无缺口）；但 H 的成功/终止仍需按 B-V5-2 修正。
- `native_sample_id` 三元组（refresh_index/sample_sim_time_s/produced_at_actor_step）由 `validate_native_sample_id` 强制；recorder 另存 actor_step。
- **config SHA 绑定**：`validate_pairing` 要求 physics/reward/termination/code/weight/stream SHA 并在 `validate_branch_pair` 比对；但 `case_record()` 只写 `evidence_contract_sha256`，不写这些配置 SHA。建议把 pairing 的 shared identity SHA 绑进 case_record（O-V5-3）。

## 4. 分支配对（通过）

- `branch_execution_plan`：75 对、150 条；每个 case 恰一 real + 一 plane；两分支 `sample_streams`/`command` 完全相同；operation_id 不同（map_branch 只进操作标识）。
- `stream_id` 的派生不含 map_branch；manifest/run-plan 哈希校验通过。
- `pre_intervention_digest`/`assert_branches_identical_until`：同前缀摘要相等；改前干预帧某字段即 ValueError；改 end_index 之后不影响；可独立复算。

## 5. CLI dry-run（阻断）

- `--dry-run`（both）：`frozen_cases=75`、`h_series_cases=6`、`selected/total trajectories=150`、`branch_pairs=75`、`env_batches=8`、3 个 S2 terrain keys、6 个 legacy H keys；与冻结清单/运行计划一致。
- **B-V5-3**：① 输出未含冻结运行计划的 **30 批**（run_plan["batches"]=30），只有 env_batches=8；② `--max-envs` 仅校验 >0，`--max-envs 200` 被接受（env_batches=1），未执行 20 环境上限。
- 非 dry-run：退出码 1，明确提示需 Q021 预留+GPU 窗口；无 `--allow-physics` 等旁路；dry-run 未启动物理。

## 6. Q027 `g_level`（行为通过；声明阻断）

- 独立手工序列：`t0=0`、`t0+1=0.04`、`t0+24=0.96`、`t0+25=1.0`、其后 1；`first_positive=t0+1`、`first_full=t0+25`；首帧不预记 0.02/0.04。
- 失效：满值后条件失败同帧归零并解除起点；重入新 `t0`，序列重来（实测 t0=31、32→0.04、56→1.0）；中断/重启计数正确。
- 平地=1；平台起步以首个有效样本为 t0（g=0 后按 25 步 ramp）；楼梯条件未满足时为 0；条件用投影 XY（无接触），腾空不清零；每 episode 新 gate。
- 快照：`version='q027_ramp_total_0p5s'` 可往返载入；pre-Q027 版本载入报错。
- 同一帧：`s2_rewards.reward_terms` 的 pitch 项 `-5·g·gx²` 与超速项 `-2·g·max(v_heading−c_exec−0.15,0)²` 共用同一 `g`（实测 g=0.25 时两项同比例）。
- **B-V5-1（阻断，文档/合同）**：`s2_state` 模块 docstring 仍写 “holds 0.5 s then ramps linearly to 1” 且 “G_LEVEL_RAMP_STEPS … provisional”，与 Q027（0.5 s 即全部 ramp、无额外 hold、非 provisional）矛盾；代码与快照版本已正确，仅声明未更新。
- 观察：`g_level_report` 只给 first_positive 相对四足入台的延迟（Q027 要四个时刻+相对延迟分别记）；`duty_cycle` 用条件成立帧（含 t0 的 g=0），`max_off_run` 只统计条件失败帧，需明确是否即 §10.2 的 g 占空比/关闭时长定义。

## 7. 输入/预算未变与旧 dry-run

- `S2_DIAG.json` `b990052c…`、run plan `bc9ac8ba…`、H manifest `1b0ad352…` 与 V4.1 快照一致；预算仍 263,888,370（S2.0 429,822），未被 T2 改动。
- 本轮 dry-run 输出保存在 `verification/dry_run_r5`、`dry_run_r5_200`，未覆盖任何既有 dry-run。

## 8. 判定

- **通过**：地形/表面查询、分支配对、证据 schema、S2.0 状态机与 Q027 数值行为。
- **阻断**：B-V5-1（g_level 文档仍写 hold+provisional）、B-V5-2（H 成功条件非 v07）、B-V5-3（dry-run 缺 30 批、env 上限未强制）。
- **待裁定**：Q-V5-1 duty/max-off 定义；Q-V5-2 四个 g 时刻的相对延迟是否全部单列；Q-V5-3 H 复测终止路径归属。

机器可读明细见 `logs/a2_v09/s2_0/verification/terrain_env_audit_20261122_r5.json`。
