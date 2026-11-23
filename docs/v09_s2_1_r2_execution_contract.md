# S2.1-r2 执行合同（Q035）

- 时间：2026-11-22
- 依据：SDS v1.2 §2.4/§2.5/§10.1、DSH-Q035、`docs/v09_s2_1_r1_invalidated.md`

## 1. 来源与初始化

| 项 | 值 |
| --- | --- |
| 入口权重 | `logs/a2_v09/s1_seed42_r3/checkpoints/s1_least.pt` |
| 入口 SHA256 | `efa9caa8ce99e2cdff967212063e198b7ae68a6415727083b1db027f615ead24` |
| 加载内容 | 规定 Actor/Critic、继承 std、冻结 normalizer（`7ac07230…`） |
| 优化器 | S2.1 全新 Adam，LR 按 SDS 入口 3e-5（KL 调整在 [1e-5,1e-4]） |
| 环境/采样器/优化状态 | 全部重置；**不继承 r1 模型或任何训练状态** |
| attempt id | `s2.1-r2`（与 r1 不同） |
| 输出目录 | `logs/a2_v09/s2/s2_1_r2`（与 r1 不同） |

## 2. 学习预算（Q035 §2）

| 项 | 值 |
| --- | --- |
| S2.1 学习上限 | 800,000,000 |
| r1 已消费（含在途保守结算） | **20,299,776** |
| r2 可用 | **779,700,224** |
| r2 完整 rollout | **31,726**（31,726 × 24,576 = 779,698,176；累计 799,997,952，余 2,048） |
| 检查点 | 每 204 rollouts（约 5,013,504 样本） |
| 累计账本 | `learning_S2_1` 累计已消费 20,299,776 + r2 有效学习；不给 r2 重开 8 亿 |

## 3. 拒绝链（Q035 §5）

- 最新检查点指针、`--resume`、门禁/终验/晋级/导出：**拒绝整个 r1 链条**（四个检查点 SHA 见作废记录）。
- r1 候选成绩、派生报告、导出一律不得用于任何判定。
- S2.0 无梯度诊断与独立正确冻结来源的对照不受牵连。

## 4. 前置条件（全部满足才可启动）

1. P0–P2 修复完成并通过 CPU 测试（episode 跨 rollout 连续、GAE last_value、§5.4 平地锚定、样本对齐、异常结算、S2 状态机/终止、命令链、真实观测量）；
2. 独立验收按 Q035 §4 用手算参照签字（20 项、+2/+8/−10、g_level、DECEL/HOLD、登阶、平台直启、动作对齐、v07 差异）；
3. 修复后的实际执行栈复核 512 资源准入（试跑不计入 r2）；
4. 台账存在、probe/admission 有效、`learning_S2_1` 余额 ≥ 31,726×24,576。

## 5. 启动命令

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
/home/xxs/anaconda3/envs/a2/bin/python scripts/train_a2_v09_s2.py \
  --execute --num-envs 512 --course S2.1 \
  --fallback-evidence logs/a2_v09/s2/probe/1024-r2/probe_report.json \
  --device cuda:0 \
  --ledger logs/a2_v09/s2/ledger.jsonl \
  --probe-report logs/a2_v09/s2/probe/512-r5/probe_report.json \
  --admission-evidence logs/a2_v09/s2/admission/s2_1_512_r1.json \
  --output-dir logs/a2_v09/s2/s2_1_r2 \
  --max-rollouts 31726 \
  --checkpoint-every-rollouts 204 \
  --attempt-id s2.1-r2
```

## 6. 监测

- 每检查点核 SHA、schema 11、`ledger_cursor` 与台账一致、std 范围、课程元数据；
- 每 1 亿样本按 SDS 触发门禁节点（与 Q035 的"每 1 亿门禁"一致），末尾去重；
- 出现非有限值/接触溢出/OOM：停并保留证据，不自动改配方。
