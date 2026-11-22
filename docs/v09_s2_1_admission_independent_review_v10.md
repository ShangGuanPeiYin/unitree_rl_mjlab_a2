# V10 独立复核：512 资源准入（开训前最后一道）

- 复核角色：A2-S2-VERIFY（独立于实现方）
- 复核依据：DSH-Q033（commit `c122666`）、SDS §2.3、既有 V8/V9 结论
- 复核方式：**纯 CPU、只读**；未运行 GPU/物理/训练，未修改实现/测试/SDS/证据/ledger/checkpoint，未做 git 写入
- 结论：**REJECT ADMISSION（暂不可写入 admission）**
- **admission review SHA：不提供**。当前 512 probe 的 `independent_copy=false`，且 ledger/512 训练配置尚有未闭合项；给出 SHA 会绕过这些前置。修复后需重新提交 V10 复核。
- 输出：本报告 + `logs/a2_v09/s2_0/verification/admission_audit_20261122_r10.json`

## 0. 结论摘要

| 项 | 独立结果 |
|---|---|
| 512 `timings_s` | **50 条**，合计 275.39699656376615 s，与 `elapsed_s` 一致 |
| 512 资源 | peak **4521.75 MiB ≤7168**；host free **14864.24 MiB ≥4096** |
| 512 eval/video/restore | eval **100×9472**；video **1×200**；`restore_ok=true`，`restore_checks=1` |
| 512 异常 | OOM/nonfinite/contact_overflow 全 false；`failure_reasons=[]` |
| 512 independent_copy | **false**（要求 true）→ **阻断** |
| 512 `ProbeResult.admitted()` | **false**，failures=`['independent_copy']`；`choose_resource` 返回 `selected=None` |
| 1024 失败证据 | `1024-r2` OOM、peak 5396.76、0 measurements、`failure_reasons=['oom']`；1024-first 顺序合规 |
| probe 台账 | probe1024 spent **0**；probe512 **2,332,928**；overage **261,064**；overage 与失败 partial 不相符，存在重复/未对账 |
| 512 训练默认 | CLI 默认 16,276 rollouts /49,152 samples /checkpoint102；512 应为 **32,552 /24,576 /204**；当前会只训 399,998,976 样本 |
| 观测合同/预算/resume/门禁 | CPU 测试 **14 passed**；观测合同、16,276×49,152=799,997,952、resume、36 组/18-of-20 均通过 |
| admission | **不可写** |

## 1. 512 probe 逐项复核

来源：`logs/a2_v09/s2/probe/512-r4/probe_report.json`、`512-r4.log`。

| 字段 | 值 | 判定 |
|---|---|---|
| `num_envs` | 512 | 符合 fallback 选择 |
| `peak_mib` | 4521.75146484375 | ≤7168，通过 |
| `host_free_mib` | 14864.23828125 | ≥4096，通过 |
| `warmups` | 5 | 通过 |
| `measurements` | 50 | 通过 |
| `timings_s` | 50 条，合计 275.39699656376615 | 与 elapsed 一致 |
| `evaluation_envs` / `steps` | 100 / 9472 | 通过 |
| `video_envs` / `steps` | 1 / 200 | 通过 |
| `restore_ok` / `checks` | true / 1 | 通过 |
| `oom/nonfinite/contact_overflow` | false/false/false | 通过 |
| `failure_reasons` | [] | 表面通过 |
| **`independent_copy`** | **false** | **不合格（要求 true）** |
| `admitted`（报告字段） | true | **与实际判定不一致** |

### 1.1 independent_copy 阻断（B-V10-1）

代码 `s2_resources.REQUIREMENTS['independent_copy'] = True`，`ProbeResult.admitted()` 明确检查该字段：

```text
ProbeResult(... independent_copy=False).admitted() -> False
failures -> ['independent_copy']
choose_resource([1024 OOM, 512]) -> selected=None
```

但 `run_probe()` 的 `failure_reasons` 没有加入 `independent_copy`，所以 probe 脚本把 `admitted` 写成了 `true`；`write_admission()` 只检查报告里的 `admitted is True`，因此**会接受这个不合格 probe**。这是准入链上的一个真实 bug，不是文档瑕疵。

必须满足其一后才能签：

1. 512 probe 用 `--independent-copy` 重新完成，并在报告中写 `independent_copy=true`；同时确认其语义（当前 docs 只说“记录调用参数”，尚未证明独立副本）；或
2. 有明确裁定允许 `independent_copy=false`，并把该裁定写入 admission 依据；否则按代码 REQUIREMENTS 不得准入。

## 2. 1024 失败证据与 fallback 合规

`logs/a2_v09/s2/probe/1024-r2/probe_report.json`：

- `num_envs=1024`、`admitted=false`、`oom=true`、`peak=5396.76025390625`、`measurements=0`、`restore=0`、`failure_reasons=['oom']`。
- `1024-r2.log` 首行：`Warp CUDA error 2: out of memory (in function wp_cuda_graph_launch)`。
- 1024-r1 的 `_sample_tiles` 签名错误已由 `f862ee5` 修复；r2 进入真实构造后 OOM，故“先 1024、后 512”的顺序合规。
- 1024 未产生 control step；reserve 3,709,184 已 release，spent 0。
- 512-r4 是在 1024 OOM 后的 fallback；**顺序本身通过**，但 512 的 independent_copy 不合格，因此不能凭此 fallback 完成 admission。

## 3. 台账：probe/overage

| 项 | 数值 | 说明 |
|---|---:|---|
| probe1024 committed | 0 | reserve/release 全退 |
| probe512 committed | 2,332,928 | 2,332,928/2,332,928 用满 |
| 其中 r1 partial | 24,576 | 失败尝试 |
| 其中 r2 partial | 122,880 | 失败尝试 |
| 其中 r3 partial | 122,880 | 失败尝试 |
| 其中 r4 successful | 2,062,592 | 成功 512 运行 |
| failures partial 合计 | 270,336 | 已计入 probe512 |
| `probe_512_retry_overage` | 261,064 | 额外独立 commit |
| overage 与 partial 差 | 9,272 | 未解释 |
| matrix net | 130,235 | 不变 |
| H committed | 3,242 | 原1,754+重跑1,488 |
| debug | 11,905 | q031 platform/stairs 增量 |
| total net | 2,739,374 | 含 probe/overage |
| envelope | 264,169,434 | ≤400,000,000 |

**B-V10-2**：

- probe512 的失败尝试 partial 已经计入 2,332,928（r1+r2+r3=270,336），成功 r4 只占 2,062,592；因此失败尝试并不是“0 控制步”。
- `probe_512_retry_overage` 又额外 commit 261,064，没有对应 refund/reclass；`270,336−261,064=9,272` 无法由“3 次失败尝试 + 1 次部分 1024（0 步）”解释。
- 若按“失败重试从重试额度出、成功运行从探测额度出”，应把失败 partial 从 probe512 移出/冲销，而不是再加一笔 overage；当前口径至少存在 **261,064 的重复计费风险**。
- envelope 264,169,434 本身 ≤400m，但这是建立在该 overage 上的；overage 未对账前，不能把预算视为闭合。

## 4. admission 前置（观测合同/预算/resume/门禁）

### 4.1 已通过

- **观测合同**：`S2TrainEnv.OBSERVATION_CONTRACT` 为 `proprio45 / history5×45 / heightmap3×40×30 / route_error3 / privileged35 / critic211`；`assert_v07_contract` 在构造时校验 control/physics dt、decimation、action scale、joints、map shape/grid/refresh；512 probe 实际构造/训练成功。
- **预算算术**：1024 配置 `16,276 × 49,152 = 799,997,952`；`s2_train.rollout_plan(512)` 为 `32,552 × 24,576 = 799,997,952`。
- **resume**：`test_checkpoint_roundtrip_and_resume_keeps_lr_std` 通过；probe `restore_ok=true`。
- **门禁计数**：36 组、每组20例、阈值18/20；相关 CPU 测试通过。
- `pytest -q tests/test_v09_s2_1_train.py`：**14 passed**。

### 4.2 未满足

1. **independent_copy=false**（B-V10-1）；且 `write_admission` / `run_probe` 对该字段有漏洞。
2. **overage 261,064 未对账**（B-V10-2）；不得凭当前台账签 admission。
3. **512 训练默认仍是 1024 口径**（B-V10-3）：
   - `scripts/train_a2_v09_s2.py` 默认 `max_rollouts = S2_1_MAX_ROLLOUTS`（16,276）；
   - `S2LearningBudget` 默认 `samples_per_rollout = 49,152`；
   - checkpoint 默认 102 rollouts。
   - 512 应为 **32,552 rollouts、24,576 samples/rollout、204 checkpoint cadence**；否则 16,276×512×48 = **399,998,976**，只到 800m 上限的一半。用户所述“16,276 rollouts /799,997,952”只对 1024 成立，和 512 fallback 不一致。
4. **H/debug 口径**：当前实际 H 3,242、debug 11,905、总 2,739,374；与当前台账一致（含 H 重跑和 q031 输入采集），matrix 130,235 正确。
5. **无 admission review SHA**：本 V10 拒绝签发；不得用本报告文件名/内容冒充 admission SHA。

## 5. 结论与下一步

**当前不能写 admission，也不能启动训练。**

按优先级：

1. 重新跑 512 probe（带独立副本证据），或取得明确豁免；修复 `run_probe` 把 `independent_copy` 计入 `failure_reasons`，并让 `write_admission` 拒绝 false。
2. 对账 `probe_512_retry_overage`：应当 reclass/refund 还是新增额度，给出一笔不重复的账；overage 不能与 probe512 partial 同时计费。
3. 在 512 执行路径上改/显式传 `--max-rollouts 32552`、`samples_per_rollout=24576`、checkpoint204；确认学习样本仍为 799,997,952。
4. 保留 1024 OOM 证据；512 fallback 顺序本身通过，但 independent_copy/overage/训练配置未闭合前不签。
5. 上述修复后重新提交 V10 复核；届时才可给 admission review SHA。

本报告不提供 admission SHA：`admission_review_sha256 = null`。
