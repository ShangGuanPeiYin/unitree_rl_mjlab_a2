# V11 独立复核：S2.1 门禁前置（冻结病例、dry-run、L40/6400 缓存、checkpoint）

- 复核角色：A2-S2-VERIFY（独立于实现方）
- 复核范围：`S2_GATE_A/B` 冻结病例、门禁执行器 dry-run、L40 缓存计划、schema11 checkpoint、6400/Q034 口径
- 方式：纯 CPU、只读；未运行 GPU/物理/训练，未修改实现/测试/SDS/证据/ledger，未做 git 写入
- 结论：**阻断（待 Q034 与缓存执行）**；门禁 dry-run 计划本身通过，但当前不能开跑 A/B 全门禁
- 输出：本报告 + `logs/a2_v09/s2_0/verification/gate_preflight_audit_20261122_r11.json`

## 0. 结论摘要

| 项 | 结果 |
|---|---|
| A/B 冻结病例 | **通过**：720/套、36 组×20、组成 9 flat + 12 stairs + 12 platform + 3 L40；manifest/file SHA 匹配；重新生成逐字节一致 |
| case key/命名空间 | **通过（命名空间）**：A/B 独立 suite 盐；A∩B=0、A/B∩S2_DIAG=0；numeric seed 仍共用42，若要求“整数 seed 不重叠”则**待澄清** |
| 门禁 dry-run | **通过（计划）**：36 组/720/1,620,000；义务可读；`physics_started=false`、`ledger_written=false`；6400/L40 缺失状态正确暴露 |
| L40 缓存计划 | **通过（计划）**：s1_least efa9、3 组/60/219,220、20 envs/batch；代码含 receipt/逐例/汇总/拒绝覆盖；**实际缓存未生成** |
| checkpoint v1 | **通过**：schema11、required metadata 无缺失、entry/teacher efa9、cursor 5,013,504=204×24,576、std范围/role counts/序列完整 |
| 6400/Q034 | **阻断/待裁定**：builder 禁用6400，但 collector 直接 `--reference 6400_flat` 仍接受并走§3.2迁移；gate executor 不校验缓存 source SHA，存在旁路 |
| 门禁开跑 | **不可**：L40 cache 未生成；6400 被 Q034 挂起且未完全封锁；flat/L40 组无法完整聚合 |

## 1. 冻结门禁病例

### 1.1 文件/清单 SHA

| 套 | manifest SHA | file SHA |
|---|---|---|
| A | `1ea91343afc22ced4c349e2704f828ab21527c6425bc056477e69c42242ab69a` | `98842a079945b55180b772c2809dc7bfd610b8581b5348dbfe38a5ceedc3448f` |
| B | `9af3f7c93b613dc24c2a02c2296594c0cf92efff7adb4c1880766881cf38d2ff` | `ecdbf5c42a31528f80eedbee141a763689685948b944cfa3557068cc23093056` |

独立复算：

- `validate_gate_manifest` 对 A/B 均返回 True；
- `build_gate_manifest` 重新生成 A/B，manifest SHA **逐字节等于冻结值**；
- 720 case/suite，36 group×20；无重复 case SHA；
- group 组成：
  - flat：9（walk×3 + stop_instant×3 + stop_ramp_2s×3）；
  - stairs：12（10/20 级 ×0.8/4.0 m ×0.3/0.5/0.6）；
  - platform：12（2/4 m 顶高 ×0.8/4.0 m ×三速）；
  - L40：3（三速）。
- `case_key_fields` = `stage,suite,task,stair_count,geometry,speed,stop_mode,force_family,case_id`；逐例 `case_key` 与字段一致，`case_sha256` 可复算。
- A/B case SHA 交集 **0**；A/B 与 `S2_DIAG`（75 case）交集 **0**。

### 1.2 seed/命名空间

- gate stream_id 含 `S2_GATE_A/B` suite 盐，A/B spawn stream 不同；训练不读 gate 清单，final 也未复用 gate case SHA。
- **numeric seed 仍为42**（`s2_spec.SEED=s1_spec.SEED`）；gate/training 共用整数 seed，但 stream 命名空间独立。
- 若父会话“seed 不重叠”指**整数 seed 互斥**，这一点**待澄清**；若指 namespace/case 不共用，则通过。

## 2. 门禁 dry-run

执行：

```text
scripts/run_a2_v09_s2_gate.py --dry-run --suite gate_A \
  --course S2.1 --checkpoint logs/a2_v09/s2/s2_1/s2_1_000005013504.pt
```

独立读取 `/tmp/v11_gate_dryrun/S2.1_gate_plan.json`：

- groups **36**、cases **720**、control_samples **1,620,000**、per group45,000；
- 义务读取：
  - `gate_admissions_up_to_40` limit173,940,000，spent/reserved0；
  - `l40_milestone_cache_gate_A` limit219,220，spent/reserved0；
- `original_6400_cache.status = missing_requires_new_6400_flat_cache`；
- `l40_cache.status = missing_requires_l40_cache_gate_A/B`；
- `physics_started=false`、`ledger_written=false`。

**判断**：

- dry-run 计划正确，缺失缓存没有被静默补造；状态暴露正确。
- 但 **gate A/B 全量 execute 仍不可**：没有 L40 cache，flat 9 组需要 6400 reference；L40 3 组需要 L40 cache。dry-run 只证明规划，不证明门禁可执行。
- 门禁 dry-run 没有加载 checkpoint；checkpoint schema11 由 §4 单独验证。

## 3. L40 缓存计划与结构

独立执行：

```text
scripts/build_a2_v09_s2_gate_caches.py --cache-kind l40 --suite gate_A \
  --dry-run --output-dir /tmp/v11_l40_plan
```

输出：

| 字段 | 值 |
|---|---|
| source | `logs/a2_v09/s1_seed42_r3/checkpoints/s1_least.pt` |
| source SHA | `efa9caa8ce99e2cdff967212063e198b7ae68a6415727083b1db027f615ead24` |
| groups | 3 |
| cases | 60 |
| control_samples | 219,220 |
| envs_per_batch | 20 |
| batches | 3 |
| physics/ledger | false/false |

- 与 §6.1.1 “同一 A/B 病例、S1 冻结模型”一致：L40 cache 使用 s1_least，不是训练学生或6400迁移模型。
- 缓存结构（代码审计）：
  - `reference_cache.json`：schema、suite、manifest SHA、source/model SHA、groups（cases/successes/rows/P10/P50/P90/failure_classes）、`case_files`；
  - `cases/<case_sha256>.json`：逐例结果；
  - `cache_receipt.json`：plan、model SHA、manifest SHA、case SHA list、budget、cache SHA、case_files；
  - 已存在 case/cache/receipt 用 `FileExistsError` 拒绝覆盖。
- **实际缓存尚未生成**：`logs/a2_v09/s2/gate_reference` 无 L40 cache；门禁 dry-run 的 `missing_requires_l40_cache_gate_A/B` 是真实状态。L40 开跑前必须先执行 cache build（GPU/台账另批）。
- 文档 `docs/v09_s2_1_gate_execution.md` 仍写 L40 使用 6400 迁移模型，与 builder 代码 `s1_least.pt` 冲突；**代码符合本次要求，文档需订正**（R-V11-2）。

## 4. checkpoint v1（schema 11）

文件：`logs/a2_v09/s2/s2_1/s2_1_000005013504.pt`，SHA `59cf153d510d605c8f2c718baeb9cb1864dd7d4da89444bfa846c5261d715665`。

`s2_checkpoint.load_checkpoint` 通过：

- outer `schema=11`、stage S2、SDS v1.2 `09fa…`、source/entry `efa9caa8…`；
- metadata 所有 `REQUIRED_METADATA` 字段存在，missing=`[]`；
- `entry_weight.sha256 = teacher_source.sha256 = efa9caa8…`；
- `policy_std` 12 维，min **0.1748897**、max **0.2504865** ∈[0.05,1.0]；
- `role_assignment`: 512 env，counts 103/102/205/51/51，和 `build_role_plan(512)` 一致；
- `gate_ab_sequence="gate_A"`、`gates_passed=[]`、`gate_uses=0`、`final_uses=0`；
- `ledger_cursor`: learning_samples **5,013,504**、rollouts **204**；204×24,576=5,013,504，文件名一致；
- `ledger_run_id="a2-v09-s2-q021-r1"` 与当前 ledger run id 一致；
- 当前 ledger learning spent 已到5,726,208（233 rollouts），是检查点之后的训练推进；**checkpoint cursor 在其保存时刻一致，不是与当前尾值相等**。

## 5. 6400/Q034（阻断）

### 5.1 已做对的部分

- `build_a2_v09_s2_gate_caches.py --cache-kind 6400` **明确拒绝**：

```text
6400 gate cache is intentionally disabled pending DSH-Q034
```

- 门禁 dry-run 正确显示 `original_6400_cache=missing_requires_new_6400_flat_cache`。

### 5.2 仍存在旁路（阻断/待裁定）

1. `scripts/collect_a2_v09_s2_gate_reference.py` **仍接受** `--reference 6400_flat`；独立 dry-run 直接返回成功计划（9组/180例/234,180样本）。其执行分支使用：

```python
V09ActorCritic.migrate_6400_checkpoint(...)
```

即 **§3.2 的 40×30 迁移模型**，不是“原6400”的48×32网络。builder 的禁用只挡住统一入口，**没有挡住 collector 直调**。
2. `run_a2_v09_s2_gate.py` 的 `--original-6400-cache` 只校验 schema/suite/manifest/group，**不校验 source SHA/是否为迁移模型**；一个由 collector 迁移路径生成的 cache 可被门禁消费。
3. 因此 Q034 裁定前 **6400 路径没有被完整禁用**；这不能作为“原6400”门禁对照。flat 9组在 Q034 前不能合法运行。

**处理建议**：在 collector 入口也对 `reference=6400_flat` 做 Q034 拒绝；gate executor 对 `source_sha256`/模型类型做显式校验；Q034 裁定前不生成/不消费任何6400 cache。

## 6. 发现清单

| ID | 级别 | 内容 |
|---|---|---|
| P-V11-1 | 通过 | A/B 720/36×20、组成、case key、manifest/file SHA、确定性重算、A/B 及 S2_DIAG 无 case 交集 |
| R-V11-1 | 待澄清 | 命名空间独立；numeric seed 仍共用42，若要求整数 seed 不重叠则需另定 |
| P-V11-2 | 通过（计划） | gate dry-run 36/720/1,620,000；义务/缺失缓存状态正确；physics/ledger false |
| P-V11-3 | 通过（计划） | L40 cache s1_least efa9、3/60/219,220、20 envs/batch；结构/receipt/拒绝覆盖代码完备；实际 cache 未生成 |
| P-V11-4 | 通过 | checkpoint schema11 字段完整；entry/teacher/cursor/std/role/gate序列通过 |
| B-V11-1 | 阻断/待裁定 | 6400 Q034 未完全封锁：collector 直调仍接受6400_flat并使用迁移模型；gate 不校验 cache source SHA |
| B-V11-2 | 阻断/待执行 | L40 cache 缺失、6400 cache 缺失；门禁 A/B 全量 execute 无法完成 flat/L40 组 |
| R-V11-2 | 待澄清/文档 | `docs/v09_s2_1_gate_execution.md` 仍称 L40 用6400迁移，与代码 s1_least 冲突 |

## 7. 结论

- 门禁病例冻结、组成、case key、A/B 独立性、确定性重算：**通过**。
- dry-run 计划、义务读取、缺失状态：**通过**；但只是计划，不能当作门禁可执行。
- L40 缓存计划：**通过**；实际缓存未生成，gate 开跑前必须执行。
- checkpoint v1：**通过**，无字段缺失；cursor 与保存时台账一致。
- **6400/Q034：阻断**。统一 builder 禁用不等于完整禁用；collector 直调和 gate 的 cache 路径仍可走§3.2迁移。Q034 裁定前不得生成或消费6400 cache。
- 因此 **门禁 A/B 当前不可全量开跑**；先完成 L40 cache、Q034 裁定与6400封锁，再复核 dry-run/execute。
