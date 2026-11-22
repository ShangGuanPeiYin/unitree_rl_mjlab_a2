# A2 v0.9 S2 执行状态与交接（2026-11-22）

SDS：v1.2 `09fa2d4c…`；分支 main；执行主对话 DSH「A2继续训练」。

## 1. 已完成（有证据、已独审）

| 项 | 证据 |
| --- | --- |
| S2.0 诊断 150 条（S2_DIAG × real/plane） | `docs/v09_s2_0_diagnostic_final.md`，`matrix-final-v7`，V9 复核 |
| S2.0 核心结论 | 40 级窄台 real 15/15 登顶后前倾摔倒（top 窗 43–55 帧全低头）；宽 4 m top 0/15（登阶失败）；platform real 宽 0.8 0/15、宽 4.0 15/15、plane 30/30；窄台外缘有效格 97.68% clip −1 |
| H 六例 | 6/6 completed，`coverage` 全 `not_verified`（Q033） |
| Q031 同帧双输入 | 平台/楼梯代表病例反事实报告；`pending`，`not_proven` |
| 资源准入 | 1024 OOM 留证；512 独立副本 probe admitted（peak 4521.75 MiB）；V10' review SHA `97ff4184…`；admission 已写 |
| 门禁/终验框架 | S2.1 36 组×20×A/B（720/套）、S2.2 33 组、S2.3 51 组已冻结；27 组终验 2,700 例 dry-run 通过 |
| 对照缓存 | L40 A 生成中（s1_least，60 例）；original-6400 路径 dry-run 通过（234,180/套），迁移模型已禁用 |
| 并发安全闸 | 训练中拒绝缓存生成（实测） |

## 2. 已作废

- `s2.1-r1`：825 rollouts / 20,275,200 样本 + 在途保守 24,576 = **20,299,776**；四个检查点 SHA 见 `docs/v09_s2_1_r1_invalidated.md`；原因：训练误用 v07 奖励、rollout 每轮 reset 等实现缺陷（Q035）。

## 3. 未完成（阻塞签字/开训）

1. 真实 S2 状态机/终止/质量判据替换 V07 全链（P1 剩余）；
2. V13 要求的证据：13 用例手算对照、两连续 sampling block shadow run、GAE/auto-reset 逐环境、锚定真实梯度、动作同帧对齐、四类异常结算、r1 拒绝测试；
3. L40 A/B 与 original-6400 A/B 缓存全量（L40 A 进行中，约 3.3 h 余）；
4. `s2.1-r2` 正式训练（31,726 rollouts，约 62–65 h）；
5. S2.1 门禁 A/B → S2.2（48,828 rollouts ≈ 90 h）→ 门禁 → S2.3（48,828 ≈ 90 h）→ 门禁；
6. 27 组终验（2,700 例）与 `S2_WALK_STOP_ACCEPTED`/失败报告。

## 4. 账本（本机 `logs/a2_v09/s2/ledger.jsonl`）

| 义务 | spent / limit |
| --- | --- |
| learning_S2_1 | 20,299,776 / 800,000,000 |
| s2_0_matrix_two_branches | 130,235 / 420,390 |
| h_series_baseline_supplement | 3,242 / 9,432 |
| s2_0_pipeline_debug | 11,905 / 20,000 |
| resource_probe_512 | 2,332,928 / 2,332,928 |
| probe_512_retry_overage | 261,064 / 261,064 |
| probe_512_independent_copy_r1 | 2,323,656 / 2,323,656 |
| l40_milestone_cache_gate_A | 1,554 / 219,220 |
| l40_milestone_cache_gate_A_rerun | 1,561+ / 219,220（进行中） |
| envelope | 269,035,966 / 400,000,000 |

## 5. 恢复执行入口

1. 等 L40 A 完成 → 校验 `reference_cache.json`/`cache_receipt.json`；
2. 完成 P1 + V13 证据与签字；
3. 启动 `s2.1-r2`（见 `docs/v09_s2_1_r2_execution_contract.md`）；
4. 门禁 A/B（见 `docs/v09_s2_1_gate_runbook.md`）→ S2.2 → S2.3 → 27 组终验。
