# V11'' 独立复核：Q034 原6400路径与 L40 门禁缓存

- 复核角色：A2-S2-VERIFY（独立于实现方）
- 复核依据：DSH-Q034（`802bd9b`）、修复提交 `fa0f26d`/`cdbbf17`、V11' 报告
- 方式：纯 CPU、只读；未运行 GPU/物理/训练，未修改实现/测试/SDS/证据/ledger，未做 git 写入
- 结论：**仍为阻断（BLOCKED）**
- 输出：本报告 + `logs/a2_v09/s2_0/verification/gate_preflight_audit_20261122_r11rr.json`

## 0. 结论摘要

| 项 | 结果 |
|---|---|
| collector 迁移旁路 | **已封堵**：`--reference 6400_flat` 直接拒绝，rc=1 |
| gate 6400 cache 校验 | **字段通过**：source SHA、model_type、3×48×32、migrated=true 拒绝；action consistency 未校验 |
| Q034-A 原始6400 dry-run | **计划通过**：9组/180/234,180、20 env、model_type original、obs 3×48×32 |
| Q034-A 原始6400 execute | **阻断**：collector 把 `env._observations`（student 40×30）喂给原始48×32 Teacher，实际会抛 ObservationError |
| L40 cache | **阻断**：实际 cache 未生成；逐例字段缺 start/last_valid/failure_class；collector 有 reset 出生点风险；gate 不校验 L40 source/字段；20-env 仅存在于 plan，实际串行 |
| checkpoint | **通过**：schema11、字段完整、cursor 204×24,576 与保存时台账一致 |
| 文档 seed | **通过**：已写明 seed42 不是流标识；L40 写 s1_least |
| 文档 6400 命令 | **仍错**：仍指向 `build... --cache-kind 6400`（该入口拒绝），未改为新 original collector |

## 1. collector 拒绝路径与 cache 身份

### 1.1 迁移旁路已拒绝（通过）

独立执行：

```text
collect_a2_v09_s2_gate_reference.py --reference 6400_flat --suite gate_A --execute
```

返回 rc=1，消息：

```text
DSH-Q034: migrated 40x30 6400 is not the original 6400 reference;
this bypass is disabled (use the Q034-A original old-map path only)
```

### 1.2 gate 6400 cache 校验（字段通过）

`run_a2_v09_s2_gate.py::_load_flat_cache` 现在检查：

- `source_sha256 == 30ed23ba16f7e490afca001fbbecadb4a11ff4b11733a5ae7ac7d0c841f9a8b2`；
- `model_type == 'original_6400'`；
- `observation_shape == (3,48,32)`；
- `migrated is True` 拒绝；
- schema/suite/manifest/groups。

**残留**：未校验 `action_consistency.passed`，也未校验 `model_sha256`/实际权重；旧迁移 cache 只能靠上述字段拒绝。当前 `gate_reference` 中未发现旧迁移 cache，因此“历史目录标无资格”没有实际对象；若后续发现，需补标记。

## 2. Q034-A 原始6400：dry-run通过，execute阻断

### 2.1 dry-run

```text
collect_a2_v09_s2_gate_original_6400.py --suite gate_A --dry-run
```

返回：

- 9组/180例/**234,180**；
- `source_sha256=model_sha256=30ed23ba…`；
- `model_type=original_6400`、`observation_shape=[3,48,32]`、`map=native_v07_48x32`；
- `envs_per_batch=20`、batches=9；
- action consistency 声明、`physics_started=false`、`ledger_written=false`。

### 2.2 execute 的实际 teacher 输入错误（阻断）

实际 collector 代码：

```python
obs = env._observations(update=True)   # returns student 3x40x30
...
action = teacher(tensor_obs)           # Frozen6400Teacher requires 3x48x32
```

`V09S1Env._observations` 返回 **student_observation**（40×30）；teacher observation 另存在 `env.teacher_observation` / `observation_pair()[0]`。原始 `Frozen6400Teacher` 的 `_validate` 要求 3×48×32；CPU 直接调用得到：

```text
ObservationError: forward: observation 'heightmap' must be [B,3,48,32], got [2,3,40,30]
```

因此 **Q034-A execute 在第一步就会失败**；当前 action consistency 只用 zeros 3×48×32 对同一模型重载，不能发现该 bug。

### 2.3 合同/批量/预留

- S2 flat 9×20 case 的 spawn/streams/command/stop timing/timeout 原样保留；flat_schedule 与 S2 case 的 timeout/ramp/hold 独立核对一致。
- 20 env batch 已使用 `V09S1Env([...20], ...)`，批量不是问题。
- 预留：代码按 **每 group** `hard_timeout*20` 预留，不是一次性 234,180 的整套原子预留；总量上界234,180正确，但“原子预留/仅实际”仍不严格。
- 实际 commit=`steps*20`；batch 中部分 env 先终止时仍按最大 steps×20 计，可能高估实际控制步。Q034-A“仅结算实际”需要逐 case 步数。

## 3. L40 cache

### 3.1 实际 cache 不存在

当前 `logs/a2_v09/s2/gate_caches/gate_A/B/` 只有 `l40_gate_*_plan.json`，没有 `reference_cache.json`、`cases/`、`cache_receipt.json`。门禁 dry-run 的 missing 状态正确，但 L40 门禁 execute 仍不可。

### 3.2 字段不满足 Q034-B

`collect_a2_v09_s2_gate_reference.py` 的 L40 record 仍是：

```python
net_distance = env.last.route_x - case.spawn.local_position_xy[0]
```

没有显式 `start_x`/`last_valid_terminal_x`；没有逐例 `failure_class`（仅 group 聚合）；没有把 timeout/engineering failure 分列。`env.last.route_x` 在 S2 auto-reset 终止行是 reset 出生点，可能把出生点当终止点；Q034-B 明确要求最后有效终止位置、排除 reset 出生行。

### 3.3 gate L40 校验缺失

`_load_l40_cache` 只校验 schema/suite/manifest/groups；**没有校验 source_sha256=efa9caa8、model_type/s1_least、逐例字段/P10/P50/P90**。`_aggregate` 只读 `cached['successes']/passed`，不核对 L40 的 P10/P50/P90 或 source。L40 cache 可能被错误 cache 消费。

### 3.4 向量化

- 计划/文档写 20 envs/batch；实际 collector 是 **单 env 逐 case 串行**，20-env 向量装配尚未实现。
- 这**不造成语义错误**：串行生成 219,220 样本仍可得到正确 cache；当前真正的阻断是“没有 cache + 字段不合规 + gate 不校验”。若采用串行 build，最小可行修复是直接运行 collector（训练结束后 GPU 窗口）并生成 A/B cache；若坚持 20-env，需补实现和 CPU/GPU 装配测试。
- 文档已给出 20 env 约1.5–2GB、必要时降4 env 或等训练结束的显存安全说明；可作为过渡方案。

## 4. checkpoint v1（通过）

`s2_1_000005013504.pt` SHA `59cf153d…`：

- schema11，`load_checkpoint` 通过，required metadata missing=[]；
- entry/teacher `efa9caa8…`；std 0.17489–0.25049；role counts 103/102/205/51/51；
- `ledger_cursor` 5,013,504/204 = 204×24,576；当前 ledger 233 rollouts（5,726,208），为 checkpoint 之后的推进；无错配。

## 5. 文档与 seed

- `v09_s2_1_gate_execution.md` 已写：
  - “`seed=42` 是固定整数种子参数，不是流标识；case SHA 由 suite/role/case_id/channel 派生”；
  - L40 使用 `s1_least.pt`（3×40×30）。
- **但 6400 生成命令仍用 `build_a2_v09_s2_gate_caches.py --cache-kind 6400`**，该入口仍明确拒绝；应改为 `scripts/collect_a2_v09_s2_gate_original_6400.py`。这是操作级阻断（照文档命令无法生成 cache）。

## 6. 可执行顺序

### 现在就能做（CPU/无 GPU）
1. 修 `collect_a2..._original_6400.py`：teacher 输入改用 `env.teacher_observation`（3×48×32）；在真实 S2 flat 首批执行中做 action consistency 检查。
2. 修 L40 collector 字段：start、last_valid_terminal、signed net、failure_class/timeout/engineering，排除 reset 出生行。
3. 补 gate L40 cache 校验：source=efa9caa8、逐例字段、P10/P50/P90、不退化判定。
4. 改文档 6400 命令到 original collector。
5. 重跑 dry-run/静态检查；不占 GPU。

### 需训练结束/GPU 窗口
6. 生成 6400 原6400 A/B cache（各234,180）和 L40 A/B cache（各219,220）；不要与训练并行。
7. 重新 V11'' 复核 cache identity；然后才能门禁 execute。

## 7. 结论

- 迁移 collector 旁路已封堵；gate 6400 cache 字段校验通过。
- 但 **Q034-A original 6400 execute 当前不可运行**：teacher 被喂 student 40×30；这是核心阻断。
- L40 cache 未生成、字段不满足、gate 不校验；L40 门禁 execute 仍不可。
- checkpoint 通过；docs seed 已写，但 6400 命令仍指向错误入口。
- 因此 **V11'' 仍为阻断**；先按§6顺序修复，再复核。
