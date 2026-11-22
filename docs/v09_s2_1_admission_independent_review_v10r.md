# V10' 独立复核：512 admission 复核通过

- 复核角色：A2-S2-VERIFY（独立于实现方）
- 复核依据：DSH-Q033（`c122666`）、SDS §2.3、V10 阻断项修复
- 复核方式：纯 CPU、只读；未运行 GPU/物理/训练，未修改实现/测试/SDS/证据/ledger，未做 git 写入
- 结论：**ADMITTED（可写入 admission）**
- 复核报告：本文件；admission review SHA 为**本文件最终内容的 SHA256**（见 §8；写入 admission 时使用该 64 hex）
- 复核对象：`512-r5/probe_report.json` (`bdb02d67bb4f3da0548cce6972865ccabff34b0d473a4f037eaa012f9d697125`)、`512-r5.log`、`1024-r2`、ledger、`train_a2_v09_s2.py` dry-run、`s2_resources` 校验代码

## 0. 结论摘要

| 项 | 结果 |
|---|---|
| 512-r5 50 次测量 | **通过**：50 条 `timings_s`，合计 278.962424 s，与 elapsed 一致 |
| peak/host | **通过**：4521.75 ≤7168；14043.97 ≥4096 |
| eval/video/restore | **通过**：100×9472、1×200、restore_ok=true |
| independent_copy | **通过**：true；entry source before/after/copy SHA 均为 `efa9caa8…`；state before/after `7ac07230…`；process PID 1928488 / ppid 1928479，argv 含 `--independent-copy` |
| failure_reasons | [] |
| 1024 fallback | **通过**：`1024-r2` OOM、peak 5396.76、0 measurements；先 1024 后 512 顺序合规 |
| 512 训练 dry-run | **通过**：32,552 rollouts ×24,576 = 799,997,952；checkpoint 204；cap 800,000,000 |
| 台账 | probe512 limit 2,332,928 + overage 261,064 + r5 2,323,656 = **4,917,648**；overage 是 r4 实际超出 probe512 限额的部分，不是重复；无 spent 的 `probe_512_rerun` 已有 no-op 注解；envelope 268,816,746 ≤4 亿 |
| 观测合同/预算/resume/门禁 | CPU `tests/test_v09_s2_1_train.py` **14 passed**；合同/上限/resume/36 组 18-of-20 通过 |
| admission | **PASS**；`probe_report_failures=[]`，`ProbeResult.admitted()=True`，`choose_resource` selected=512 |

## 1. 512-r5 逐项复算

| 字段 | 独立值 | 判定 |
|---|---|---|
| `num_envs` | 512 | 通过 |
| `peak_mib` | 4521.75146484375 | ≤7168 |
| `host_free_mib` | 14043.97265625 | ≥4096 |
| `measurements` | 50 | 通过 |
| `timings_s` | 50 条，sum 278.9624241895508 | 与 `elapsed_s` 一致 |
| `evaluation_envs` / `steps` | 100 / 9472 | 通过 |
| `video_envs` / `steps` | 1 / 200 | 通过 |
| `restore_ok` / `checks` | true / 1 | 通过 |
| `oom/nonfinite/contact_overflow` | false/false/false | 通过 |
| `failure_reasons` | [] | 通过 |
| `independent_copy` | true | 通过 |
| `admitted` | true | 通过 |

- `512-r5.log` 是同一 probe 的输出；其中 `timings_s`、eval/video/restore 与 report 一致。
- report SHA：`bdb02d67bb4f3da0548cce6972865ccabff34b0d473a4f037eaa012f9d697125`，与父会话一致。

### 1.1 independent copy 证据

```text
entry_weight.source_sha256_before = efa9caa8ce99e2cdff967212063e198b7ae68a6415727083b1db027f615ead24
entry_weight.source_sha256_after  = efa9caa8…（同）
entry_weight.probe_copy_sha256     = efa9caa8…（同）
entry_state.sha256_before          = 7ac07230c93d720624ac255c4b1d35e27a814c93c0f657d8b6c73c9cc8dab6fb
entry_state.sha256_after_reload    = 7ac07230…（同）
process.pid/ppid                   = 1928488/1928479
process.argv                       = ... --independent-copy ...
process.independent_process        = true
```

- 独立用 CPU 重新加载 `s1_least.pt` 计算 state SHA，得到 `7ac07230…`，与 report 一致。
- 源文件、探针副本、before/after 均一致；source 未变。
- `process.independent_process` 字段仍由 probe 代码自报，但 PID≠PPID、argv/launcher 独立且包含 `--independent-copy`，本次可接受。

## 2. 校验代码路径

`s2_resources.probe_report_failures()` 独立重算全部 requirements，不信任报告的 `admitted`：

- peak/host、warmups、measurements、**timings 条数**、restore、evaluation_envs/steps、video_envs/steps；
- `independent_copy`、entry source before/after、probe copy SHA、entry actor state before/after、process evidence。

用 512-r5 复算：

```text
probe_report_failures(r5, num_envs=512) -> []
ProbeResult(512-r5).admitted() -> True
choose_resource([1024-r2, 512-r5]) -> selected=512
```

`write_admission()` 现在调用 `probe_report_failures()`；`validate_admission()` 也重新校验 probe 文件和 requirements，并校验 admission 的 entry weight SHA。独立用 r5 做临时 admission 写入+validate：**write ok / validate True**（临时路径，不写正式 admission）。

## 3. 1024 失败与 fallback

- `1024-r2`：`oom=true`、peak 5396.76025390625、`measurements=0`、`restore_checks=0`、`failure_reasons=['oom']`；log 为 `Warp CUDA error 2: out of memory (wp_cuda_graph_launch)`。
- `1024-r1` 的 `_sample_tiles` 签名错误已修复；r2 进入真实图构造后 OOM，故 **1024 先、512 后**的顺序合规，没有以 512 代替未修实现。
- 512-r5 是独立副本 probe，不是对 1024 的伪造。

## 4. 台账对账

### 4.1 r4/overage

r4 实际（父会话给出口径，独立按组件相加）：

```text
5 warmup ×24,576          = 122,880
50 measured ×24,576       = 1,228,800
eval 100×9,472             = 947,200
video 1×200                = 200
restore 1×24,576           = 24,576
r4 total                   = 2,323,656
```

probe512 限额2,332,928 = 失败 partial（r1 24,576 + r2 122,880 + r3 122,880 =270,336）+ r4 在 probe512 内可计部分 **2,062,592**。

overage261,064 = r4 实际2,323,656 − r4 内可计2,062,592。

所以：

```text
probe512 2,332,928 + overage 261,064
= failed partial 270,336 + r4 actual 2,323,656
= 2,593,992
```

这**不是**重复计费：overage 是 r4 超出 probe512 限额的实际部分；先前 V10 把“partial + overage”误读为重复，本次按完整分解纠正。

### 4.2 r5 与 unused

- `probe_512_independent_copy_r1` commit **2,323,656**，与 r5 实际 full run 一致。
- probe-related 合计：270,336（失败 partial）+2,323,656（r4 full）+2,323,656（r5）= **4,917,648**。
- `probe_512_rerun` 是误追加的 **unused** 预留：reserve/commit 均0，spent 0；已有 ledger 事件 sequence623 注解 `unused duplicate extension ... kept as capacity only, spent=0`。它增加 envelope 2,323,656，但未消耗控制样本；envelope **268,816,746 ≤400,000,000**，不影响准入。
- 当前实际总 spent：matrix130,235 + H3,242 + debug11,905 + probe5122,332,928 + overage261,064 + r5 2,323,656 = **5,063,030**。父会话的2,739,374是 r5 前的旧快照。
- ledger 的 `probe_512_retry_overage` reason 文案仍写成“3 failed attempts ...”，与实际 r4 超额口径不一致；金额/归属正确，建议追加一条说明事件，不要改金额或重复扣费。

## 5. 512 训练口径

独立运行（CPU）：

```text
python scripts/train_a2_v09_s2.py --dry-run --num-envs 512 \
  --device cpu --fallback-evidence logs/a2_v09/s2/probe/1024-r2/probe_report.json
```

得到：

```text
training_plan.max_rollouts            = 32,552
training_plan.samples_per_rollout     = 24,576
training_plan.checkpoint_every_rollouts = 204
training_plan.learning_cap            = 800,000,000
32,552 × 24,576 = 799,997,952
```

代码已用 `training_plan` 设置 `S2LearningBudget(samples_per_rollout)`、`max_rollouts` 和 checkpoint cadence；`learning_S2_1` 限额800,000,000、reserved 0，launch 前会检查 effective cap ≤ learning limit。**通过。**

## 6. admission 前置

| 前置 | 结果 |
|---|---|
| 512 probe requirements | 通过（`probe_report_failures=[]`） |
| 1024 OOM fallback | 通过 |
| independent copy/entry source/state/process | 通过 |
| `write_admission` 重校验 requirements | 通过（代码路径 + 临时写入/validate） |
| 观测合同 | 通过（OBSERVATION_CONTRACT；assert_v07_contract；512 probe 实际构造/训练） |
| 学习预算 | 通过（800M；32,552×24,576=799,997,952；reserved 0） |
| resume | 通过（`test_checkpoint_roundtrip_and_resume_keeps_lr_std`） |
| 门禁计数 | 通过（36 组、20 例、18/20；CPU 测试） |
| 512 rollout/checkpoint 口径 | 通过（32,552/24,576/204） |
| 未用 rerun | spent 0，已有 no-op 注解；只占 envelope，不占 spent |

- CPU 测试：`pytest -q tests/test_v09_s2_1_train.py` → **14 passed**。
- 未执行：正式训练、门禁实跑、GPU 训练；本 review 只签资源准入与训练前置。

## 7. 仍建议的非阻断修正

1. ledger 的 `probe_512_retry_overage` reason 文案应改为“r4 实际2,323,656 − probe512内2,062,592 =261,064”，并说明不是 failed attempts 重试；金额不变。
2. `probe_512_rerun` 已有 no-op 注解，建议再标明“envelope-only, no spend”并保持 spent0。
3. `process.independent_process` 是由 probe 自报字段；本次 PID/argv/launcher/entry SHA 足够，但未来可改为由父进程比对 PID，进一步硬化。

以上不阻断 admission；不得据此重算 spent 或重复扣费。

## 8. Admission review SHA

**review SHA = 本文件 `docs/v09_s2_1_admission_independent_review_v10r.md` 的 SHA256**。

写入 admission 时使用该 64 hex；对应 `--independent-review-path` 指向本文件。不要使用 V10 的 `admission_review_sha256=null` 或旧报告。

## 9. 结论

- 512-r5 的测量、eval、video、restore、资源与 independent copy 证据通过。
- 1024 OOM 证据和 512 fallback 顺序通过。
- overage 已按 r4 实际超额解释，不重复；unused rerun spent0 且有注解；envelope 268,816,746 ≤4 亿。
- 512 训练口径已修正为 32,552/24,576/204，800M 上限一致。
- **ADMITTED：可写入 admission。**
