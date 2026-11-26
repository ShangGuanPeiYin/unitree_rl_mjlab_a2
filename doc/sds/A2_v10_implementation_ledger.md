# A2 v1.0 实施账本

规范：SDS_A2_v1.0_DualTeacher_DAgger_PPO_v1.1.md，SHA256 eb95905ed24bf0674f94a1b02dab02256ee11e40a20ed1cd96eb24daeb1b49a0。
基线提交：36c08f4f575288927f8c789f1e1377420cd7785b；按明确授权在main实施，不建worktree。

## 文件归属及任务

1. 主代理：spec/cases/evaluation/checkpoint/orchestration、CLI、日志、Git。
2. integration_inventory：新geometry/tasks/env/contacts及环境测试。
3. learning_stack：新model/teachers/replay/dagger/ppo/rewards及学习栈测试。
4. review_preflight：独立只读审核（不得批准自己修改的实现）。

旧v09、已有脚本和未提交测试均不纳入本次暂存。规划线程单独落盘的新SDS及其日志由主代理统一提交。

## 实施前接口核对

| 任务间接口 | 生产者/消费者 | 核对结果 |
|---|---|---|
| 真实观测 | env → teachers/model/rollout | proprio45/history5×45/map3×40×30/route3/privileged35，快照clone，只推进一次 |
| 动作后遥测 | env → rewards/evaluation | command/phase/g_level锁存动作前，其余为动作后；同帧事件只结算一次 |
| 终点质量 | evaluation.EpisodeMetrics → env | normal终点先判质量，再确定成功奖励；不完整窗口失败 |
| 聚合数据 | runner → DAgger | 当前rollout先更新，后reservoir入库；CPU FP32快照 |
| 暂停恢复 | runner/checkpoint → orchestration | 先末状态bootstrap/GAE/update，退出GPU再评估，新物理回合恢复 |

## 已确认授权和裁定

- 规划线程已确认完整v1.1文件；其“规划模式/未发送”文字为历史状态，非继续等待条件。
- 正式训练前必须双栈资源探测与教师A/B准入，失败不绕过。
- 学习预算196608000；评估/probe另计。资源优先1024，失败512，均失败停止。
- 旧奖励接线不复用：旧stable_stand仅看零指令、joint limit填零、raw仅Z力及post-step门控均不符合新版。

## 验证记录

- spec/cases测试：初次因目标模块不存在失败；实现后3项通过。
- evaluation缺窗及配对覆盖测试：初次因模块不存在失败；实现后4项通过。
- Torch既有jit.script弃用警告1条。
- 真实物理、资源、教师准入、训练及终验尚未执行，未声称行为改善。
