# A1 D1 50轮试训额度防护独立工程审核（2026-11-27）

结论：对下列文件版本的生产入口、恢复/费用逻辑与定向CPU证据，未发现阻断性缺口，trial guard工程通过。此结论仅覆盖一次D1调整试训的工程限制，不是资源、教师能力、112例诊断、AB或正式训练准入。资源v1真实失败的历史结论不变，本次没有新增物理、GPU、full3GB测试或训练；没有修改受审源码。完整源码须待v2接线完成后由主代理统一冻结。

## 授权边界与实际接线

| 要求 | 实际生产接线 | 独立结论 |
|---|---|---|
| 原D1与总学习预算不增加 | `spec.py:21`保留D1=24576000，D2=49152000，D3=73728000，PPO=49152000，总196608000；runtime worker在原cap上取min(trial.cap) | 通过；试训最多2457600，计入原D1账本，不新增学习额度 |
| 限定D1/1024/50/2457600 | `trial_guard.py:17`–27严格匹配四参数，要求显式trial-id/canonical guard/ledger | 通过；任一trial参数非None即进入该严格校验，不能只传半套配置 |
| 固定身份不换run赠额 | `trial_guard.py:23`–39将canonical guard解析路径、ledger绝对路径、trial-id、SDS及cap持久绑定，O_EXCL创建、文件和目录fsync；已有不一致即拒绝 | 通过；换output目录不会改变额度，换guard/ledger/trial-id被拒绝；损坏文件会失败而不会重新赠额 |
| charged=executed+uncertain、不退款 | `checkpoint.py:110`–207 Q1锁定append-only reserve/execute journal；runtime:162 reconcile；ledger charged作为限制依据 | 通过；未持久确认的物理步恢复为uncertain并继续charged，不能按checkpoint回退退款 |
| 仅余完整rollout才开始 | `trial_guard.py:48`–49要求charged+1024×48≤2457600且retained<50；runtime:212循环与collector:budget_limit=cap双重限制 | 通过；余49152允许，余48128拒绝，没有半轮凑预算 |
| trial CP省参数不能无帽续跑 | runtime:164–171加载CP后要求matching trial identity；不带trial参数而extra含trial即拒绝 | 通过；同cap/同ledger/同trial身份才能恢复；checkpoint原SDS/code/physics/N校验仍在 |
| 首次完整CP保存失败不学习 | runtime:209–211在首collect前保存trial_initial；save_worker保留模型/Adam/RNG/replay/env/费用及trial extra，checkpoint保存沿用完整校验 | 通过；CPU disk-full注入验证collect/update均0且无成功receipt。初始replay为空是零更新状态，完整指checkpoint字段与状态完整，不声称已填满 |
| 旧initial恢复不把历史更新冒充保留成果 | trial.restore只读取CP retained_completed_updates；外账historical_updated另列；runtime:247–249以retained生成round/updated_samples | 通过；50轮后从旧initial恢复得到retained=0/updated_samples=0，历史updated=2457600，费用不退且不再更新 |
| 50边界确实保存 | runtime:228–249保留原周期保存及边界硬链接复用/保存，receipt只在成功返回后写入 | 通过；CPU走50次真实Adam＋小replay/完整checkpoint，最终CP可加载且SHA一致；final-save失败不生成成功worker_result |
| 不伪装为门禁通过 | trial.status将pre_gate_adjusted_trial=true、teacher_admitted=false、accepted=null、training_accepted=false、gate_reached=false，写入CP extra与结果 | 通过；50轮只代表试训停止边界，不准许正式AB或后续阶段 |
| 单worker串行 | `scripts/run_a2_v10.py:33`–35在worker外持有`gpu_execution_lock`；`gpu_lock.py:15`–21使用flock | 通过于生产CLI入口；子进程实测锁跨父进程边界互斥并在释放后可重取。直接Python runtime.worker函数自身不加锁，审核不将该内部API称作可并发入口 |
| 默认非trial行为保持 | runtime diff仅新增可选trial分支，None默认沿用旧while/cap/保存/结果路径；原logging与protocol测试回归 | 通过；没有改课程、教师配方、奖励、损失、学习率、随机化或终止规则 |

## 实际测试证据

独立执行（非复述实现者报告）：

```bash
/home/xxs/anaconda3/envs/a2/bin/python -m pytest tests/test_v10_trial_guard.py tests/test_v10_runtime_task_logging.py tests/test_v10_protocol.py -q
```

实际结果：**29 passed，1个上游torch.jit.script弃用warning，55.83 s**。其中trial10项，原logging5项，原protocol14项。本次CPU seam以小型模型/6条数据和capacity=1 replay替代真实1024环境，模拟全rollout费用；模型Adam及checkpoint保存/加载真实执行。该测试不能冒充50轮真实物理或满replay资源验证。

trial覆盖：固定绑定/参数拒绝；reserved、executed未mark、已mark三种崩溃恢复；uncertain不退与完整rollout余量；正常50停止/CP恢复/重复恢复不增额；旧initial恢复retained与历史成果分离；省略trial参数恢复trial CP拒绝；初始保存失败无学习；最终保存失败无成功receipt。例：1024 uncertain＋49×49152 charged后余48128，不再开始第50轮；重复恢复保持49 retained。protocol包含真实CPU子进程flock互斥测试，logging包含非trial保存及恢复行为回归。

未运行完整项目测试、full3GB、GPU、真实资源探测、112例诊断或试训；最终全套及v2生产接线由主代理统一验证。仅在资源和112诊断均通过并满足新裁定的前置条件后，才可由主代理启动已授权D1原预算内50轮试训；本报告本身不授予跳过前置门禁的能力。

## 受审文件身份

- `src/tasks/velocity/v10/trial_guard.py`：SHA256 `b6e7dad6715d0f4c4387e399cd5b3f34e7a9023d14080d605318417cabc074e1`。
- `src/tasks/velocity/v10/runtime.py`：SHA256 `00677d34add3c195061d4167b0eff2611ffa8007a58c07556b4d0fecbae7f043`。
- `scripts/run_a2_v10.py`：SHA256 `5cf82b67497953658a7a066ce937a5348888f82940c13e0faee622af674976d6`。
- `tests/test_v10_trial_guard.py`：SHA256 `962a619004aca1ca14f1406c3d2ac00b315e6e68e76b2906140ff61dbdd98c3f`。
- `src/tasks/velocity/v10/checkpoint.py`：SHA256 `12620a96dd69db42ad4c5769519cb89d7c65367b99dda38e581dfcb99675c737`。
- `src/tasks/velocity/v10/spec.py`：SHA256 `8c0691b8aa053d3318059b05bdab1c68fb1df1195b3a1a7709e88f6d1fb0a841`。
- `src/tasks/velocity/v10/gpu_lock.py`：SHA256 `7c997056590795f3bb96617a003f0f3fc7f24625741c0e986ad7957c2c34c197`。

以上SHA仅绑定本次实际读取/测试版本。若后续v2接线改变受审逻辑，应按差异复核，不能将本结论无条件沿用到新字节。变更日志及提交推送由主代理统一处理。
