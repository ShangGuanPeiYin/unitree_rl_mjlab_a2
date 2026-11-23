# S2 执行状态（2026-11-23，r2 启动后）

## 1. 签字与基线

| 项 | 值 |
|---|---|
| 代码基线 | `5114024`（main，已 push；实现最终 SHA `bfb2550`，签字后仅文档提交） |
| 独立验收结论 | **GRANTED**（A2-S2-VERIFY，`docs/v09_s2_q036_independent_verification.md` §12） |
| Q035/Q036 链 | 签字完成，允许从 `s1_least.pt` 直接启动全新 r2 |
| 入口权重 | `logs/a2_v09/s1_seed42_r3/checkpoints/s1_least.pt`，SHA256 `efa9caa8ce99e2cdff967212063e198b7ae68a6415727083b1db027f615ead24` |
| SDS | v1.2，SHA256 `09fa2d4cdd858914c46a51fd52670c20cb20f2b6f3e76e6d266054574f64ccbe` |
| 资源准入 | 512 env（1024 OOM），`logs/a2_v09/s2/admission/s2_1_512_r1.json`，probe `probe/512-r5` |

## 2. r2 运行参数

- 命令（后台 nohup，pid 2128402）：
  `scripts/train_a2_v09_s2.py --execute --num-envs 512 --course S2.1 --device cuda:0 --attempt-id s2.1-r2 --output-dir logs/a2_v09/s2/train/s2.1-r2 --ledger logs/a2_v09/s2/ledger.jsonl --probe-report logs/a2_v09/s2/probe/512-r5/probe_report.json --admission-evidence logs/a2_v09/s2/admission/s2_1_512_r1.json --fallback-evidence logs/a2_v09/s2/probe/1024-r2/probe_report.json --trace-dir logs/a2_v09/s2/train/s2.1-r2/trace --trace-every 5000 --trace-max-rollouts 3`
- 日志 `logs/a2_v09/s2/r2.log`；checkpoint 目录 `logs/a2_v09/s2/train/s2.1-r2/`（每 204 rollouts 一个）。
- 规模：31,726 完整 rollouts × 24,576 = 779,698,176 样本；预计约 50–62 h。
- 预算：`learning_S2_1` 已结算 20,299,776，余 779,700,224；r2 累计后 799,997,952/800,000,000，余 2,048（按 Q037 更正算术）。

## 3. 质量与 top 口径（Q036/Q037）

- 训练/门禁/终验共用 `src/tasks/velocity/v09/s2_quality.py`；门禁 `s2_acceptance.quality_reasons` 与 `metrics.shared_quality_window` 均委托同一核心。
- `top_arrived` = 屏蔽 v07 终止前捕获的原始安全登顶信号；四足入台 +0.4 m 仅是 §5.2 门控；楼梯 `D_post` 零点为 `top_x`。
- 完成步：flat_walk 1000、flat_stop instant 1400 / ramp 1500（1001/1401/1501 仅硬保护）；停车请求 400 巡航区间后，ramp k=1..100，第 100 为 0。
- 证据：`logs/a2_v09/s2/q036/`（final2）、`logs/a2_v09/s2/v13_shadow_final/`（fresh 512-env Run A）、`logs/a2_v09/s2/v13/v13_cpu_evidence.json`（`18427c6d…`，all_passed）。

## 4. 后续顺序

1. r2 完成或按关卡保存点触发：S2.1 门禁 A/B（36 组，用已验收的 L40/6400 缓存与 `learning_S2_2` 等台账游标）。
2. S2.2 训练（33 组）、S2.3 训练（51 组）及各自门禁。
3. 27 组终验 + H 六例复现 → `S2_WALK_STOP_ACCEPTED` 或明确失败报告。
4. 每阶段实现与独立验收分离；新歧义交 Codex 规划对话裁定；代码和文档随改随 push。

## 5. 未覆盖/非阻塞余项

- 自然轨迹中的“质量合格停车”和“登顶后 +8”未出现（S1 入口策略不达标/物理失败）；已按 Q036 C3 用生产路径可控边界覆盖，r2 期间可在同一 `S2TrainEnv` 用受控初态补采自然样本。
- S1/S2.0 metrics 的楼梯窗口默认用去抖支撑代理；需要严格等价时传 `four_feet_entry`。
- GPU 20 项奖励手算（Run B/D）与真实门禁/终验轨迹尚未执行（r2 门禁阶段补）。
