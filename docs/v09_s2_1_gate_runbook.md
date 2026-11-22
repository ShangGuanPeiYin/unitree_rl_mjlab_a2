# S2.1 门禁执行手册（Q034 口径）

## 0. 前置

| 项 | 状态/来源 |
| --- | --- |
| 门禁病例 | `logs/a2_v09/s2/gate_cases/v1/S2_GATE_{A,B}.json`，manifest `1ea91343…`/`9af3f7c9…`，各 720 例 |
| 门禁执行器 | `scripts/run_a2_v09_s2_gate.py --suite gate_A\|gate_B --course S2.1 --checkpoint <schema11>` |
| 候选检查点 | S2.1 训练产出的 schema 11 检查点（`course=S2.1`） |
| 预算义务 | `gate_admissions_up_to_40`（173,940,000）、`l40_milestone_cache_gate_{A,B}`（各 219,220）；6400 平地对照按套 234,180 |
| 台账 | `logs/a2_v09/s2/ledger.jsonl`（run_id `a2-v09-s2-q021-r1`） |

## 1. 生成参考缓存（训练结束后 GPU 窗口，按序）

1. **L40 里程碑缓存**（s1_least，原生 3×40×30）：
   ```bash
   python scripts/build_a2_v09_s2_gate_caches.py --suite gate_A --cache-kind l40 \
     --execute --device cuda:0 --ledger logs/a2_v09/s2/ledger.jsonl \
     --output-dir logs/a2_v09/s2/gate_caches/gate_A
   # gate_B 同理
   ```
   上界各 219,220，录 `l40_milestone_cache_gate_A/B`；逐例字段见 Q034-B。
2. **original 6400 平地缓存**（原始权重 + 原生 3×48×32 旧图 + v07 教师观测）：
   ```bash
   python scripts/collect_a2_v09_s2_gate_original_6400.py --suite gate_A \
     --execute --device cuda:0 --ledger logs/a2_v09/s2/ledger.jsonl \
     --output-dir logs/a2_v09/s2/gate_caches/gate_A/6400
   # gate_B 同理
   ```
   上界各 234,180；要求 `model_type=original_6400`、`observation_shape=(3,48,32)`、`source_sha256=30ed23ba…`、`migrated!=true`。
3. **禁止**：迁移模型（`migrate_6400_checkpoint`）、resize、仅校验权重 SHA 的"伪原教师"。

## 2. 门禁执行（A/B 各一次，同候选 SHA）

```bash
python scripts/run_a2_v09_s2_gate.py --suite gate_A --course S2.1 \
  --checkpoint <candidate.pt> --device cuda:0 \
  --ledger logs/a2_v09/s2/ledger.jsonl \
  --original-6400-cache logs/a2_v09/s2/gate_caches/gate_A/6400/reference_cache.json \
  --l40-cache logs/a2_v09/s2/gate_caches/gate_A/l40/reference_cache.json \
  --obligation-id gate_admissions_up_to_40 \
  --output-dir logs/a2_v09/s2/gates/s2_1/gate_A --attempt-id s2.1-gate-A-r1
```

- 每组 ≥18/20；平地 9 组同时满足**绝对**与**相对原 6400 保持**；
- L40 里程碑 3 组按**不退化**（成功数、P10/P50/P90、配对净距离、物理失败分类），不强加 18/20；
- 逐例质量判据按 §6.5；失败保留；
- 连续两次门禁（A/B 两次评估，同一候选 SHA）通过才进 S2.2。

## 3. 记录与晋级

- 门禁报告：`S2.1_gate_result.json`（逐组通过数、质量、失败分类、成本、候选 SHA、cache SHA）。
- 台账：`gate_admissions_up_to_40` 按实际结算；L40 缓存用其专用义务。
- 晋级 S2.2：`--course S2.2` 续训，额度 12 亿、512 下 48,828 rollouts，LR/std/Adam 不重置。
- 门禁失败：保留证据，按 SDS §8.2 回退规则处理，不擅自改配方。

## 4. 数据完整性

- 原始 trace/npz/inputs 只读；派生摘要版本化并标 `supersedes`；
- 所有 cache 文件带 SHA、病例 manifest SHA、模型 SHA、逐例 SHA、预算与批次；重跑拒绝覆盖；
- 门禁不得以 dry-run、迁移结果或相似病例顶替。
