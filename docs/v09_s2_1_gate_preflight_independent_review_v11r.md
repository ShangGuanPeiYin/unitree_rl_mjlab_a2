# V11' 独立复核：Q034 后的门禁前置

- 复核角色：A2-S2-VERIFY（独立于实现方）
- 复核依据：DSH-Q034（commit `802bd9b`）、V11 报告
- 方式：纯 CPU、只读；未运行 GPU/物理/训练，未修改实现/测试/SDS/证据/ledger，未做 git 写入
- 结论：**仍为阻断（BLOCKED）**；Q034 只留档，代码侧阻断项尚未实现
- 输出：本报告 + `logs/a2_v09/s2_0/verification/gate_preflight_audit_20261122_r11r.json`

## 0. 结论摘要

| 项 | 结果 |
|---|---|
| 冻结门禁病例 | **通过**：720/套、36组×20、SHA/组成/case key/确定性重算/无交集 |
| 门禁 dry-run | **计划通过**：36/720/1,620,000；义务可读；physics/ledger=false；缺失缓存状态正确 |
| checkpoint | **通过**：schema11、metadata 无缺失、entry/teacher/cursor/std/role/gate字段通过 |
| 6400 旁路 | **阻断**：collector 仍接受 `6400_flat` dry-run，执行分支仍用40×30迁移；gate 不校验 source/模型类型 |
| Q034-A 原6400 | **未实现**：没有原始30ed23ba + 原生48×32旧图 + v07教师路径的S2门禁执行 |
| L40 cache | **阻断/待执行**：实际 cache 未生成；per-case 字段/末有效位置/20-env vectorization 未闭合 |
| 文档/seed | **未更新**：仍写L40用6400迁移；无suite盐/seed42说明 |

## 1. 冻结病例（通过）

- A/B 各 **720**，36组×20；组成 **9 flat + 12 stairs + 12 platform + 3 L40**。
- manifest/file SHA 与父会话一致：
  - A manifest `1ea91343…`，file `98842a07…`
  - B manifest `9af3f7c9…`，file `ecdbf5c4…`
- `validate_gate_manifest` 通过；重新生成A/B逐字节一致。
- `case_key_fields` 正确；A∩B=0，A/B∩`S2_DIAG`=0。
- 命名空间独立；numeric seed42仍共用（文档未说明“整数seed不是重叠判据”）。

## 2. 门禁 dry-run（计划通过，执行仍阻断）

- 36组/720例/**1,620,000**；`gate_admissions_up_to_40` limit173,940,000；`l40_milestone_cache_gate_A` 219,220。
- `original_6400_cache=missing_requires_new_6400_flat_cache`；
  `l40_cache=missing_requires_l40_cache_gate_A/B`；
- `physics_started=false`、`ledger_written=false`。
- dry-run 正确暴露缺失，但没有 cache 时 flat 9组/L40 3组不能完成执行。

## 3. 6400/Q034（阻断，未修）

### 3.1 collector 旁路仍在

独立执行：

```text
scripts/collect_a2_v09_s2_gate_reference.py --reference 6400_flat \
  --suite gate_A --dry-run --output-dir /tmp/v11_6400_collector_plan
```

返回 **rc=0**，计划 9组/180例/234,180，说明 collector 仍接受 6400_flat。代码执行分支仍是：

```python
teacher = V09ActorCritic.migrate_6400_checkpoint(...)
```

即 §3.2 的 **40×30 迁移模型**，不是 Q034-A 要求的原始 **30ed23ba… + 原生3×48×32旧图**。

### 3.2 gate 不校验 cache 身份

`run_a2_v09_s2_gate.py::_load_flat_cache` 只校验 schema/suite/manifest/groups；没有：

- `source_sha256 == 30ed23ba16f7e490afca001fbbecadb4a11ff4b11733a5ae7ac7d0c841f9a8b2`；
- 3×48×32 旧图观测合同；
- 原始非迁移模型类型；
- 迁移产物“无资格”标记。

因此 collector 直调产生的迁移 cache 可以被门禁消费。

### 3.3 Q034-A 原6400路径未实现

- `Frozen6400Teacher` 类存在（原始48×32），但 gate/collector 没有把它接到 S2 平地 9×20 A/B 的 v07 物理/教师观测路径。
- 没有“病例初态/流/指令/停车/物理/终止/超时/质量窗同一合同”的原始6400执行。
- 没有 A/B 各 **234,180** 的原子预留/仅实际结算实现。
- 没有对迁移产物做“无资格”标记。
- 结论：**B-V11R-1/B-V11R-2/B-V11R-3 均未闭合**。

## 4. L40 cache（计划通过；实际与字段仍阻断）

### 4.1 计划

- `s1_least.pt` SHA `efa9caa8…`；3组/60例/**219,220**；A/B各一份；obligation `l40_milestone_cache_gate_A/B`。
- 计划 `envs_per_batch=20`、batches3；但只是计划字段。

### 4.2 实际 cache 不存在

`logs/a2_v09/s2/gate_caches/gate_A|l40_gate_A_plan.json` 存在，但没有 `reference_cache.json`、`cases/`、`cache_receipt.json`；gate dry-run 的 missing 状态正确，实际执行仍不可。

### 4.3 字段/逻辑问题

`collect_a2_v09_s2_gate_reference.py` 的 L40 record 目前：

- `net_distance = env.last.route_x - case.spawn.local_position_xy[0]`；
- 没有显式 `start/spawn` 字段、`last_valid_terminal_route` 字段；
- 没有逐例 `failure_class`，只有 group 级 `failure_classes` 聚合；
- `env.last.route_x` 在 S2 auto-reset 终止时可能是 **reset 出生点**（V8/V9 已证实 terminal trace row 回到 birth），不是“最后有效终止位置”；这会产生错误的带符号净距离。

Q034-B 要求逐例保存 case SHA/起点/最后有效终止/带符号净距离/成功/失败类别/超时与工程失败；当前代码不满足。

### 4.4 20 envs/batch

只写在 plan；collector 实际是单 case/单 env 串行循环，没有 20-env vectorized 实现，也没有显存安全的分批方案。Q034-B/父会话要求“已向量化到20 envs/batch或给出显存安全方案”，当前两者都没有。

## 5. checkpoint v1（通过）

`s2_1_000005013504.pt` SHA `59cf153d…`：

- schema11；`load_checkpoint` 通过；required metadata missing=[]；
- entry/teacher SHA 均 `efa9caa8…`；
- `ledger_cursor` 5,013,504/204 = 204×24,576；当前 ledger 已推进到233 rollouts（5,726,208），checkpoint 是其保存时快照，无错配；
- std 0.17489–0.25049 ∈[0.05,1]；
- role counts 103/102/205/51/51；gate_ab_sequence gate_A、gates_passed=[]、gate_uses0、final_uses0。
- **通过，不构成阻断。**

## 6. 文档与 seed

`docs/v09_s2_1_gate_execution.md` 仍未更新：

- 第23-31行仍命令 `collect ... --reference 6400_flat`（迁移路径）；
- 第40-41行仍称 L40 用“同一 6400 迁移模型”；
- 第84-85行仍称6400路径暂停/待Q034，未写Q034已裁定后的原6400 48×32规则。
- 没有“门禁 stream 由 suite 盐派生、case SHA 独立，整数seed42不是重叠判据”的说明。

## 7. 必须先做的顺序

1. **实现Q034-A原6400路径**：原始 `30ed23ba…`、原生3×48×32旧图、v07物理/教师观测、S2平地9×20同合同；拒绝迁移/resize；旧迁移产物标无资格。
2. **封锁旁路**：collector `6400_flat` 直接拒绝；gate `_load_flat_cache` 强校验 source SHA、3×48×32、原始模型类型；不接受迁移cache。
3. **修L40 collector**：显式 start、last valid terminal（禁reset birth）、带符号net、逐例success/failure class/timeout/engineering failure；实现20-env batch或给出显存安全方案。
4. **生成L40 A/B cache**（各219,220，仅实际结算），再跑gate dry-run/execute。
5. **更新文档**：6400/Q034路径、L40 s1_least、seed/suite盐说明。
6. 重新提交V11'复核；在此之前门禁A/B不得开跑。

## 8. 结论

- 冻结病例和checkpoint通过；门禁dry-run计划通过但缓存缺失。
- Q034后的**代码实现尚未落地**：6400 collector/gate 旁路仍存在，Q034-A原6400路径未实现；L40 cache实际未生成且字段/vectorization不满足。
- 因此 **V11' 仍为阻断**，不能开跑门禁；按§7顺序修复后再复核。
