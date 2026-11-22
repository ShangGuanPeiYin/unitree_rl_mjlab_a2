# A2 v0.9 S2.0 正式预留独立复核（V7a）

- 审核 agent：A2-S2-VERIFY（独立验收子 agent，未参与实现）
- 审核时间：2026-11-22（快照 14:52–15:02）
- 仓库/分支/HEAD：`/home/xxs/dog/unitree_rl_mjlab_a2` / `main` / `a272faa`
- 范围：正式预留 plan 的逐项独立复核 + 写入语义 + T6 物理接口快速核验；纯 CPU，未写正式台账、未跑物理/训练。
- 机器可读件：`logs/a2_v09/s2_0/verification/reservation_audit_20261122_r7a.json`

## 0. 结论

| 项 | 结论 | 说明 |
|---|---|---|
| 预留 plan 逐项 | **通过** | 逐项独立重算与 plan 完全一致 |
| plan SHA / caps | **通过** | `773b53c6…` 可由 plan 内容复算；263,888,370 / 3.2e9 / 3,463,888,370 / 429,822 |
| 去重 / 等待步 / S1 / Q025 | **通过** | 无重复项；等待项为 0（已含批上界）；S1 不复用；Q025 时限正确 |
| 1024/512 探测 | **通过** | 两档均按 §2.3 全项计；512 仅回退（`choose_resource` 已强制先 1024） |
| 写入语义 | **通过** | `reserve_stage` 一次性、cap、子额度、operation_id 唯一；`--write` 二次拒绝；预留与消费分离 |
| T6 物理接口 | **阻断（GPU smoke 前）** | 实 backend 的 S2 tile_id=-1 使 V07Env schedule 报错；driver/policy/env 接口不匹配；first_step 映射错；fake native id 偏移 |
| **总判定** | **预留可写入；GPU smoke 暂不可** | 台账 `--write` 不受 T6 阻断影响；物理部分待 T6 修复 |

## 1. 逐项独立重算（不调用实现方的逐项数值）

公式：`T_probe = ceil((2·(1.1+40·√(0.2²+0.2²)+1.2+10)/0.3 + 2 + 20 + 10)/0.02) = 9472`；
`probe(N)=N·55·48 + N·48 + 101·T`（55 热身+测量、1 恢复、100 评估、1 录像）。

| 项 | 独立值 | plan 值 | 判定 |
|---|---:|---:|---|
| probe1024 | 3,709,184 | 3,709,184 | 通过 |
| probe512 | 2,332,928 | 2,332,928 | 通过 |
| S2.0 两分支 | 420,390 | 420,390 | 通过 |
| H 基线补采 | 9,432 | 9,432 | 通过 |
| H 候选复测（3 模型） | 28,296 | 28,296 | 通过 |
| L40 A | 219,220 | 219,220 | 通过 |
| L40 B | 219,220 | 219,220 | 通过 |
| ≤40 门禁（40×S2.3 最坏 4,348,500） | 173,940,000 | 173,940,000 | 通过 |
| 三套终验（27,669,900×3） | 83,009,700 | 83,009,700 | 通过 |
| 等待/空步（显式 0，已含批上界） | 0 | 0 | 通过 |
| **非学习合计** | **263,888,370** | 263,888,370 | 通过 |
| 学习（8+12+12 亿） | **3,200,000,000** | 3,200,000,000 | 通过 |
| **stage 合计** | **3,463,888,370** | 3,463,888,370 | 通过 |
| S2.0 子额度 | **429,822** | 429,822 | 通过 |

- plan SHA：`sha256(plan - plan_sha256) = 773b53c68e065e228841a907ff7df14b7786c7c86809522158a4d46f203174d3`，与 `--dry-run`/`--write` 输出一致；`s2_ledger.validate_plan(plan)=True`。
- 物理子步：4×3,463,888,370=13,855,553,480，与 plan 一致。
- S2.3 最坏门禁独立复算：flat 234,180 + L40 stop 1,694,480 + L40 walk 721,240 + 10/20 复习 766,520 + 8m 平台 stop 932,080 = **4,348,500**；×40=173,940,000。
- 终验每套独立复算：candidate 13,249,500 + S1 13,249,500 + 6400 平地 1,170,900 = 27,669,900；×3=83,009,700。

## 2. 去重、Q025、S1

- 10 个非学习项 obligation_id 唯一、无重复；S2.0 与 H、门禁、终验、探测按对象分列。
- `waiting_empty_steps_no_extra_charge` 计 0 且标注“embedded in each batch upper bound”，批上界已含完成 slot 等待/padding，没有再叠加。
- `billing.s1_samples_reused=false`、`double_counting=false`，无 S1 旧账项。
- Q025：S2.0 平台组时限为 `2×5/v+10`（5 m 含首米、从出生算，未加成 6 m）；完整登阶 `L_arrival=1.2`、`D_post=5` 自 top；门禁/终验按 §4.2 的 10 m/5 m。独立重算 420,390 与 plan 一致。
- L40 A/B 缓存各 219,220（3 速×20 例×逐速原时限和 10,961），与门禁候选运行不重叠。

## 3. 1024/512 探测

- probe1024：`1024×55×48`（5 热身+50 测量）+ `1024×48`（恢复）+ `100×9472`（评估）+ `1×9472`（录像）= 3,709,184；含 §2.3 的 5/50/1、完整 48 步、每 rollout 20 Adam（计算负载）、独立副本、评估 100 环境。
- probe512：同式 = 2,332,928；`s2_resources.choose_resource` 已强制“先 1024、失败才 512”，512 单独传入会报错。
- 两档均计入预留；合计 6,042,112，远小于 4 亿；实际只按使用的档位承诺。

## 4. 写入语义

- `reserve_stage(plan)`：校验 plan SHA、逐项 basis/类别/唯一 id、cap、S2.0 子额度、stage 一次性；已有 envelope/obligations 时拒绝。
- operation_id：reserve/commit 均要求非空且全局唯一（重复 op 实测 `ValueError`）。
- 预留/消费：`state()` 分开 `reserved` 与 `spent`；实测 reserve 1000 后 reserved 增、commit 后 reserved 减/spent 增；envelope 不变。
- `--write`：首次写入 `written=true, envelope=263,888,370`；第二次退出码 2、`written=false, refused="S2 stage already reserved"`，幂等拒绝；`--dry-run` 与 `--write` 互斥。
- 本次只在 `/tmp` 临时 ledger 上测试 `--write`，**未创建/修改项目正式台账**。

## 5. T6 物理接口（阻断 GPU smoke）

- **实 backend 的 S2 tile_id 错误（阻断）**：`build_s2_env(case, device='cuda:0')`（无 factory）现在默认 `MjlabS2Backend`，随后 V07Env `set_schedule` 报 `ValueError: schedule tile id outside the 23-tile table`。根因：S2_DIAG 的 case geometry `tile_id=None`，`s2_terrain.spec_from_geometry` 将其解析为 `tile_id=-1`；V07Env 的 23-tile schedule 不接受 -1。这会在 GPU smoke 起点失败，需按 terrain_key 解析到合法 tile id（或补丁 schedule）。
- **driver/env 接口不匹配（阻断）**：`s2_execute.S2ExecutionDriver` 要求 `policy` callable，而 `scripts/run_a2_v09_s2_0.py --smoke` 传 `policy=None`（构造即 ValueError）；driver 调 `env.observation()`/向量化 `env.step(actions)`，而 `S2PhysicsEnv` 只有 `observe()`、单环境 `step(action)`。需统一接口或增加适配层。
- **first_step 映射错（阻断 S2.0 顺序）**：`S2PhysicsEnv` 把 `state_frame['first_step']=frame.top_arrived`，导致 fake 楼梯运行 `enter_climb=[]`（top 在 step 18 才进 CLIMB），APPROACH→CLIMB→TOP_TRANSITION 顺序不成立。应使用真实首阶信号。
- **fake native 三元组偏移（观察/测试合同）**：fake 运行 first replacement Actor step=19，而 `native_sample_id.produced_at_actor_step=21`（reset 双步导致），违反 Q026 关联键“生产步 ≤ 消费步”；生产 backend 需保证一致，fake 应修正。
- **通过项**：`build_s2_env` 提供 fake factory 时 device='cpu' 可构建；H legacy 走 `legacy_v07_success`（route 2.6 成功、2.0 不成功）；Q016 两分支在 first replacement=19 前的前缀摘要相同（`prefix_equal=True`）、`s−t=2`；`batch_table` 30 批；预留 gate 的 begin/finish 批次成本与批上界一致。

## 6. 判定与建议

- **预留 plan：通过**，数字/SHA/去重/写入语义均可信；主对话可执行 `reserve_a2_v09_s2.py --write`。
- **GPU smoke：阻断**于 T6 的 tile_id=-1、driver/env 接口、first_step 映射；建议先修这三项并补 fake native id 偏移，再开 smoke 窗口。
- 未决：真实 backend 的 tile_id→terrain_key 解析方式；smoke 的 policy 接口（真实 Actor 包装）；fake native id 索引是否仅测试用。

机器可读明细见 `logs/a2_v09/s2_0/verification/reservation_audit_20261122_r7a.json`。
