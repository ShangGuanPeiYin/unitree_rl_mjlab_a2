# S2.1 门禁执行与对照缓存

## 冻结门禁病例 v1

- A：`logs/a2_v09/s2/gate_cases/v1/S2_GATE_A.json`
  - manifest SHA `1ea91343afc22ced4c349e2704f828ab21527c6425bc056477e69c42242ab69a`
  - file SHA `98842a079945b55180b772c2809dc7bfd610b8581b5348dbfe38a5ceedc3448f`
- B：`logs/a2_v09/s2/gate_cases/v1/S2_GATE_B.json`
  - manifest SHA `9af3f7c93b613dc24c2a02c2296594c0cf92efff7adb4c1880766881cf38d2ff`
  - file SHA `ecdbf5c42a31528f80eedbee141a763689685948b944cfa3557068cc23093056`
- 每套 36 组 ×20 例 = 720 例；上界 **1,620,000** 控制样本；
  L40 3 组 **219,220**。
- A/B 使用独立 suite 盐，与训练、S2.0 diagnostic、S1 gate/final 不共用
  病例 SHA；冻结脚本拒绝覆盖。

## 6400 平地对照：Q010 不能直接复用

S1.0/Q010 已封存的 6400 gate/final 缓存绑定 **S1 冻结病例**；S2 gate
`S2_GATE_A/B` 的平地病例是独立 S2 流。实测 S1 gate_A 与 S2 gate_A 的
flat case SHA 交集为 **0**（各 180 例），因此 Q010 不能把旧缓存直接拿来
做 S2 gate 平地保持。

需要重新采集 6400 在同一 S2 平地病例上的迁移模型输出：

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
/home/xxs/anaconda3/envs/a2/bin/python scripts/collect_a2_v09_s2_gate_reference.py \
  --reference 6400_flat --suite gate_A --device cuda:0 \
  --obligation-id gate_admissions_up_to_40 \
  --ledger logs/a2_v09/s2/ledger.jsonl \
  --output-dir logs/a2_v09/s2/gate_reference/6400_flat-A-r1
```

B 套同命令，`--suite gate_B`。该缓存每套 **234,180** 控制样本，A/B
合计 **468,360**；不包含在 1,620,000/套的候选+L40 上界内，应记入
`gate_admissions_up_to_40`（或另行批准的评估额度）。

## L40 里程碑缓存

旧 L40 的 3 组也使用独立 S2 gate case；不能复用 S1.0 L40 缓存。必须在
S2.1 门禁前由验收/评估侧用同一 6400 迁移模型生成 A/B 各一份：

```bash
/home/xxs/anaconda3/envs/a2/bin/python scripts/collect_a2_v09_s2_gate_reference.py \
  --reference s1_l40 --suite gate_A --device cuda:0 \
  --obligation-id l40_milestone_cache_gate_A \
  --ledger logs/a2_v09/s2/ledger.jsonl \
  --output-dir logs/a2_v09/s2/gate_reference/l40-A-r1
```

B 套把 `gate_A` 换成 `gate_B`、义务换成 `l40_milestone_cache_gate_B`。
每套 **219,220**，A/B 合计 **438,440**。缓存必须绑定冻结 manifest SHA
和 6400 source SHA；在 S2.1 训练结束后、门禁开始前生成，避免与训练共用
GPU 窗口。

## 门禁命令

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
/home/xxs/anaconda3/envs/a2/bin/python scripts/run_a2_v09_s2_gate.py \
  --course S2.1 --suite gate_A --execute \
  --checkpoint logs/a2_v09/s2/s2_1/s2_1_<samples>.pt \
  --manifest logs/a2_v09/s2/gate_cases/v1/S2_GATE_A.json \
  --original-6400-cache logs/a2_v09/s2/gate_reference/6400_flat-A-r1/6400_flat_S2_GATE_A_cache.json \
  --l40-cache logs/a2_v09/s2/gate_reference/l40-A-r1/s1_l40_S2_GATE_A_cache.json \
  --obligation-id gate_admissions_up_to_40 \
  --ledger logs/a2_v09/s2/ledger.jsonl --device cuda:0 \
  --output-dir logs/a2_v09/s2/gate/S2.1_gate_A_r1
```

B 套替换 suite/manifest/cache/output；每套上界 1,620,000，A/B 3,240,000。
门禁报告为 `S2.1_gate_result.json`，逐组 18/20、逐例质量、失败分类、
6400 平地保持与 L40 cache 结果。

## 尚未完成

- `run_a2_v09_s2_gate.py` 当前 `--course` 只支持 S2.1；S2.2 的 33 组、
  S2.3 的 51 组冻结清单和执行器尚未实现。
- 6400/L40 cache 采集器已实现 dry-run；真实采集需 GPU 和台账预留，本次
  未运行。
