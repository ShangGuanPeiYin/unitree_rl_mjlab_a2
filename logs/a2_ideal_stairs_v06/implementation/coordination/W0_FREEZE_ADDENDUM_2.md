# W0 冻结补充 2：并入 Q002 裁决（v0.6）

- 时间：2026-11-16T15:20+08:00；DSH 主对话登记。
- 生效合同 = SDS（`1ed4bf5f…6ad60`）+ Q001 裁决（`5f3f2aa3…a4b1`）+ **Q002 裁决**（`3ebc7441f9be457cb135846366fe23926d02ec69e2232e0502d6c318ce5ccd52`，97 行）。
- 冲突以 Q002 → Q001 → SDS 顺序为准；原 SDS/W0/历史 manifest 保留，不追改旧 SHA。本补充与 `W0_MANIFEST_v3.json` 一并冻结。

## 裁决落点（实现口径）

| 编号 | 裁决 | 落点 |
|---|---|---|
| Q002-1 | F0→H：**连续两次完整门禁**；F0 generation ≥100 更新才累计；100 更新周期完整评估（`.3/.5/.8`、每档 20 例 ≥18）；最早 F0 更新 200 晋级（不是新增配额）；有效失败清 streak；评估异常按错误处理；F0 预算仍 1000；H00 自己两次门禁单独执行 | `v06/runner.py`、`v06/curriculum.py`、`v06/speed.py` |
| Q002-2 | `spec.FALL_DROP=0.50` 必需常量，events/env 直接读取、无 `getattr` 默认；下坠判据用**期望踏面**：`region=(N>0)&(local_x>=0)`；`z_expected=min(max(floor(local_x/d)+1,0),N)*h`；`below=region&(local_z<z_expected-0.50)`（严格小于）；平台区 `z_expected=N*h`；flat 与 `local_x<0` 不启用；不得改为足端接触面/平均支撑/出生高度；文档措辞更正 | `v06/spec.py`、`v06/events.py`、`v06/env.py` |
| Q002-3 | 最终三速 `.2/.4/.6` 各 100 例，全部用 **R 完整扩张** `x[-1.20,-.80] y±.10 yaw±10°`，独立均匀采样、不施加训练 ramp；L40 N40/h.20/d.20；test seed 固定 `400342` 且只加一次；case 表须显式标 `full_R_expanded` 并证明生成器使用该组参数 | `v06/evaluation.py`、`v06/runner.py` |
| Q002-4 | 冻结八类停止态：`behavior_regression`、`behavior_collapse`、`curriculum_stalled`、`budget_exhausted`、`gate_failed`、`user_requested_stop`、`monitoring_blocked`、`resource_insufficient`；另保留 `running`、`ready_for_final_test`、`complete`（`complete` 仅最终 300 例通过）；`pilot_not_applicable` 不是实际终态；工程错误保留类型化 error_type、不得伪造 `complete`；输出携带 reason/global_update/phase/substage/generation/各层 budget/触发证据 | `v06/spec.py`、`v06/runner.py`、`v06/recovery.py`、`v06/orchestration.py` |
| Q002-5 | 正常边界先保存→门禁/诊断/保持集→晋级或最终资格→再判预算；总预算与阶段同帧耗尽且无完成资格 → `budget_exhausted`（证据同时记两个耗尽）；未耗总预算而阶段/子级耗尽无合法前进 → `gate_failed` + `reason=phase_budget_exhausted`/`substage_budget_exhausted`；单次门禁未过但仍有预算 → 不立即 gate_failed；阶段末次合法晋级且总预算有余 → 提交晋级；总预算 30000 恰好取得 R 资格 → 允许**零学习**最终 300 例测试（通过 complete/否则 gate_failed + `stop_reason=final_test_failed`），不得第 30001 次学习；总预算只到中间阶段晋级仍 `budget_exhausted`；行为/停车类停机不被预算标签覆盖；不得强行跑 GPU 完成门禁 | `v06/runner.py`、`v06/orchestration.py` |
| Q002-6 | 保持集首次检查 = 严格下一个全局 200 倍数：`first_due=200*(floor(q/200)+1)`，之后每个完成 +200；每个集合独立保存 `qualified_at/next_due/last_completed/streak`；两集合都 due 都执行；资格不因退级/generation 改变而推迟；失败计数只按本集有效通过清零；checkpoint 保存调度身份、恢复补完 pending 评估、不重复累计 | `v06/curriculum.py`、`v06/runner.py`、`v06/recovery.py` |
| Q002-7 | V ramp 用 V 阶段内**已完成 PPO 更新数** `u`（进入 V 时 0），`p=min(max(u,0)/1000,1)`；u=0 `[.3,.8]`、u=500 `[.25,.7]`、u≥1000 `[.2,.6]`；不用全局 update、不因 reset/eval/checkpoint/resume 清零；episode 只在 reset 采新命令；范围变化影响之后采样并逐帧记录实际指令；V 门禁从第一天用最终三速、≥1000 后计 streak、最早 1200 晋级；R ramp 同理用 R 内已完成更新/400 | `v06/env.py`、`v06/runner.py`、`v06/speed.py` |

## 必须新增的 CPU 用例

F0：100 过/200 过→晋级；100 过/200 未过/300 过→不晋级；400 再过→晋级；1000 最后更新完成合法第二通过→晋级，否则预算停机。
最终：总预算 30000 末次取得 R 资格→零学习最终测试；有效测试未达标→`gate_failed`/`final_test_failed`，不 complete。
保持集：严格下一 200 倍数；两集合并存；不得从 q 起每 200；退级/generation 不推迟首检；恢复补完不重复。
V：resume 不重置 `u`；u=0/500/1000 三个区间边界采样。
下坠：跨阶边界、平台、flat、局部/世界平移、严格等号/nextafter。

## 不变项

奖励权重、PPO、课程预算（1000/12000/3000/6000/4000/4000）、门禁阈值（18/20、95/100）、速度、seed、30000 上限、D1 offset、22 tile 几何均不变；不恢复 R2 seed43；不 commit/push。
