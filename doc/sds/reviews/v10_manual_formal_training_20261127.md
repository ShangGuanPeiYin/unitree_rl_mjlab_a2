# 手动运行 A2 v1.0 正式训练

本入口使用本机已有运行目录及账本；不启动新试训，不创建第二份学习额度。冻结 SDS v1.1 与有效 A1 教师修订继续生效。原第 50 轮 checkpoint 保持原字节。

## 启动命令

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
bash scripts/run_sds_manual.sh --mode all --device cuda:0 --num-envs 1024 --rollout 48 --seed 42 --resume-reason '手动启动正式SDS训练'
```

只核对身份、证据和路径，不运行物理：

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
bash scripts/run_sds_manual.sh --mode check --device cuda:0 --num-envs 1024 --rollout 48 --seed 42
```

`--mode teachers` 只完成教师准入，不开始学习；再次用 `--mode all` 才生成交接并继续。

在前台终端运行，Ctrl+C 停止。再次使用相同 `--mode all` 命令恢复同一运行；不要删除账本、controller、证书或 checkpoint 来重置次数。脚本不会自动重试失败的行为门禁或最终验收；失败将返回非零退出码。

## 执行顺序及参数

1. 核验限定旧新源码的兼容证据，沿用已通过的双栈资源补证和 112 例诊断。证据继承不称为重跑。
2. 完成完整教师 A/B 准入。中断的旧评估不能视为通过，实际费用包括 uncertain 继续计入共享评估账本。
3. 全部门禁通过后才生成试训到正式训练交接凭据，恢复原模型、Adam、replay、随机状态和任务轮转，重新开始物理回合。保留已花费的 2,457,600 D1 样本。
4. DAgger：D1 上限 500 轮（已有 50 轮计入）、D2 1000 轮、D3 1500 轮。各阶段在 40%/80%/100% 绝对预算处评估，符合原规则可提前晋级；D1 首次正式门禁为累计 200 轮。
5. DAgger 最终验收通过才进入 PPO。P0 占 PPO 前 100 轮，PPO 总上限 1000 轮；按原 25%/50%/75%/100% 门禁及保持/回退规则执行。最终通过才导出部署文件。

1024 个环境、每轮 48 步。每 2,457,600 学习样本保存（对应 50 轮），阶段/门禁边界也保存。速度集合仍为 0.3/0.5/0.6 m/s；学习率仍为 3e-5，六个任务 replay 各 32768 条 FP32，微批量 256。奖励、损失、速度指令、模型和 A1 yaw 纠偏参数均未因本入口调整。预算是上限，不承诺耗尽预算一定达到行为门槛。

当前资源已经选择 1024；本恢复入口不接受改成 512，因为原 checkpoint 和资源证书都绑定 N=1024。更换 N 需要新的合格证据和预算裁定，不能用启动参数绕过。

## 日志及结果

运行根目录：`logs/a2_v10/formal_A1_20261127_approved`。

- `controller_state.json`：当前阶段、停止原因和验收状态。
- `operations/handoff_<新源码SHA前16位>/<operation>/attempt_*.log`：评估/学习进程输出。
- `learning/<course>/updates.jsonl`：逐轮损失、各任务回合和教师标签覆盖。
- `learning/<course>/*.pt`：周期及边界 checkpoint。
- `deployment/`：仅最终验收通过后生成。

正式评估共享 `logs/a2_v10/formal_A_20261127_approved/evaluation_costs.jsonl`；正式学习沿用当前根目录的 `learning_costs.jsonl`。并行启动会被锁拒绝。累计预算不足或门禁失败时停止，不删组、不缩短病例、不降低阈值。

## 验证边界

本次交付进行 CPU 工程验证和只读预检；不新增物理采样，不声称完成了正式训练或最终行为验收。新实现的最终验证及提交信息见同轮工程验收报告与 `logs/codex_changes.md`。
