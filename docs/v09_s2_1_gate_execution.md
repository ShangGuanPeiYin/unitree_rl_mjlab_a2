# S2.1 门禁执行与对照缓存（Q034）

## 冻结门禁病例 v1

- A：`logs/a2_v09/s2/gate_cases/v1/S2_GATE_A.json`
  - manifest SHA `1ea91343afc22ced4c349e2704f828ab21527c6425bc056477e69c42242ab69a`
  - file SHA `98842a079945b55180b772c2809dc7bfd610b8581b5348dbfe38a5ceedc3448f`
- B：`logs/a2_v09/s2/gate_cases/v1/S2_GATE_B.json`
  - manifest SHA `9af3f7c93b613dc24c2a02c2296594c0cf92efff7adb4c1880766881cf38d2ff`
  - file SHA `ecdbf5c42a31528f80eedbee141a763689685948b944cfa3557068cc23093056`
- 每套 36 组 ×20 例 = 720 例；上界 **1,620,000**；L40 3 组 **219,220**。
- A/B 使用独立 suite 盐；`seed=42` 是固定的整数种子参数，**不是流标识**。
  病例流由 `(generator, SDS, stage, suite, role, case_id, channel)` 派生，
  case SHA 独立；不得用“整数 seed 相同”判断病例重叠。A/B、训练、
  S2.0 diagnostic、S1 gate/final 的 case SHA 均独立。

## 6400 平地参考：Q034-A

Q010 已封存的 6400 S1 缓存绑定 S1 冻结病例；S1 gate_A 与 S2_GATE_A 的
flat case SHA 交集为 **0**，不能复用。

Q034-A 规定使用：

- 原始 6400 source：`logs/a2_ideal_stairs_v07/continue_seed42_from004817_trial01/checkpoints/stop_006400.pt`
  SHA `30ed23ba…`；
- 原生 **3×48×32** 旧图、原 Actor/归一化/历史、确定性推理；
- v07 物理/教师观测路径，S2 门禁平地 9 组×20 例，初态/流/指令/停车时序/
  物理/终止/超时/质量窗与候选同一冻结合同；
- 不得使用 §3.2 的 40×30 迁移模型、resize 或仅校验源权重 SHA；
- 动作一致性自检：同一旧图输入下与原教师路径动作逐元素一致；
- A/B 各 234,180 上界，合计 468,360，原子预留、仅结算实际。

生成命令：

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
/home/xxs/anaconda3/envs/a2/bin/python scripts/collect_a2_v09_s2_gate_original_6400.py \
  --suite gate_A --execute --device cuda:0 \
  --ledger logs/a2_v09/s2/ledger.jsonl \
  --output-dir logs/a2_v09/s2/gate_caches/6400_gate_A
```

`build_a2_v09_s2_gate_caches.py --cache-kind 6400` 仍拒绝，因为它不是
Q034-A 的原始旧图实现；统一入口不用于 6400。

B 套替换 `--suite gate_B`。缓存 `reference_cache.json` 必须声明
`source_sha256=30ed23ba…`、`model_type=original_6400`、`migrated=false`、
`observation_shape=[3,48,32]`；门禁读取时逐项复核，迁移/错误 source 拒绝。

## L40 里程碑参考：Q034-B

使用冻结 S2 入口 `s1_least.pt`（SHA `efa9caa8…`，原生 **3×40×30**）在
A/B 各 3 速度×20 旧 L40 病例上执行。每套上界 **219,220**，A/B 合计
**438,440**；义务 `l40_milestone_cache_gate_A/B`。按不退化门槛，不强加
18/20。

```bash
/home/xxs/anaconda3/envs/a2/bin/python scripts/build_a2_v09_s2_gate_caches.py \
  --suite gate_A --cache-kind l40 --device cuda:0 \
  --ledger logs/a2_v09/s2/ledger.jsonl \
  --output-dir logs/a2_v09/s2/gate_caches/l40_gate_A
```

缓存结构：

```text
reference_cache.json
cases/<case_sha256>.json
cache_receipt.json
```

逐例保存 case SHA、起点、最后有效终止位置、带符号净距离
`x_terminal − x_spawn`、成功/失败类别（含超时/工程失败）、trace；每套每
速度 20 例计算 P10/P50/P90；组内另存成功数、失败分类、配对候选-参考均值。
不使用 reset 后出生点。重跑拒绝覆盖。

## 门禁命令

```bash
/home/xxs/anaconda3/envs/a2/bin/python scripts/run_a2_v09_s2_gate.py \
  --course S2.1 --suite gate_A --execute \
  --checkpoint logs/a2_v09/s2/s2_1/s2_1_<samples>.pt \
  --manifest logs/a2_v09/s2/gate_cases/v1/S2_GATE_A.json \
  --original-6400-cache logs/a2_v09/s2/gate_caches/6400_gate_A/reference_cache.json \
  --l40-cache logs/a2_v09/s2/gate_caches/l40_gate_A/reference_cache.json \
  --obligation-id gate_admissions_up_to_40 \
  --ledger logs/a2_v09/s2/ledger.jsonl --device cuda:0 \
  --output-dir logs/a2_v09/s2/gate/S2.1_gate_A_r1
```

B 套替换 suite/manifest/cache/output；每套 1,620,000 上界，A/B 3,240,000。
输出 `S2.1_gate_result.json`，含逐组 18/20、逐例质量、失败分类、6400 平地
保持与 L40 cache 结果。

## 向量化

L40/6400 采集计划为单进程 **20 envs/batch**。L40 20 env 显存约 1.5–2 GB；
若与训练并行导致峰值 >7168 MiB，必须降到 4 env 或等待训练结束。当前
真实采集器先实现逐例；20-env 向量装配和 CPU 装配测试作为下一项，未运行
GPU。不得为降显存改物理/地图/网络。

## 尚未完成

- `run_a2_v09_s2_gate.py` 的 `--course` 只有 S2.1；S2.2 33 组、S2.3 51 组
  冻结清单和门禁矩阵尚未实现。
- S2.3 的 stop 状态机/奖励列在真实 V07 训练后端尚未接入。


## 并发安全闸

`collect_a2_v09_s2_gate_original_6400.py` 与 L40 collector 在 `--execute`
前调用 `s2_concurrency.require_safe_window`：

- 扫描 `/proc/*/cmdline`，发现 `train_a2_v09_s2.py --execute` 即视为训练
  正在运行；
- 读取当前 GPU `free_mib`，要求 ≥ 20 env 估算 2048 MiB + 安全余量
  1024 MiB = **3072 MiB**；
- 默认拒绝并提示 `wait for training window`；
- 只有显式 `--allow-concurrent` 且实测 free_mib 足够才放行，cache/receipt
  记录 `concurrent_with_training`、`gpu_free_mib`、`gpu_total_mib`、
  `estimate_mib`、`safety_margin_mib`；
- 不允许通过改物理/地图/网络或静默降并行来凑显存；降 4 env 仍需显式规则
  与实测。
