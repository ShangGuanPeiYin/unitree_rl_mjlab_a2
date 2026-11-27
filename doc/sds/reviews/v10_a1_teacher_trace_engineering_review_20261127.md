# A1教师适配与trace v2工程复核（2026-11-27）

范围：独立只读审核learning_stack新增教师适配器；实现本代理获批trace记录通道后由根代理复核。本文不能作为本代理对自身trace实现的独立签字，也不构成资源、教师能力或正式训练准入证书。

## 教师实现的独立判断

已阅读实际 `src/tasks/velocity/v10/teachers.py`、`tests/test_v10_teacher_yaw.py`，并在CPU执行该组测试。A1公式与规划约定一致：y=.2×route0；psi=atan2(route1,route2)；psi_des=clip(−atan2(.2y,max(vx,.1)),±.15)；wz=clip(.5×atan2(sin(psi_des−psi),cos(psi_des−psi)),±.15)×clip(vx/.3,0,1)。非正vx三命令全零；正vx保留，vy固定零。

正y/正yaw纠偏符号、期望角限幅、角速度限幅后乘速度门控、低速.1分母、±pi分支均与批准公式一致。BlindTeacher仅clone自身proprio并改6:9；学生proprio/history及共享存储不变，权重/normalizer不变。没有隐藏task/phase门控、状态、额外环境真值。DualTeacher混合路由保留stairs原输入与原路径，所有blind调用统一适配。teacher_identity绑定两教师权重与含实际常量/协议的canonical配置摘要；eval参考身份已复用该helper，另保留全源码身份。

真实ETH网络以手算私有命令进行CPU前向对照一致，CPU测试支持接线正确；normalizer非零统计仍不证明原训练命令范围。平台纠偏效果、完整新教师准入及短资源补充测试均尚未运行，不能宣称通过。

## trace交付与根审关联修复

`env.py`仅在既有frame追加已计算yaw(rad)、局部足坐标、raw contact、向上top力及body_hit/long_hit/attitude/route_fall/bounds/timeout独立mask；没有改变终止、奖励、观测或课程逻辑。

`trace.py`保留旧grading rows20，并另存schema v2数组：action-time原请求/私有命令/route3/yaw、teacher路由与命令有效标记，post-step yaw/feet/raw/top/masks。私有命令复用生产blind_yaw_command，未复制公式；candidate参考不假称执行教师。每例包含首终止帧，之后冻结；padding不写入病例trace。shape、finite、boolean mask、容量、row-count对齐错误均作工程错误。

根审指出初版逐帧dict和小数组持有会显著增加formal长病例内存。本代理已按要求改为TraceBatchV2(n,max_steps)：每字段一个连续预分配NumPy数组，保留源float32/float64及bool精度，使用独立计数与冻结集，不持有逐帧list/dict，不concat/stack完整轨迹，save直接使用已记录区间的view。NPZ压缩临时峰值仍须纳入真实资源监控，CPU测试不能替代该门槛。

评估仅显式diagnostic_manifest允许teachers/A/4例且强制early-stop，按4/4检查、diagnostic_only=true、不读取或写入正式cache；默认正式20例18/20门槛不变。诊断manifest的冻结/来源/SHA校验由根新diagnostic入口负责，未在本trace单元自行另造病例。

## 验证与边界

测试先RED验证缺失trace/诊断接口，随后GREEN；根审预分配要求再次RED复现接口缺失，再GREEN。最终CPU命令为：

```bash
/home/xxs/anaconda3/envs/a2/bin/python -m pytest tests/test_v10_trace.py tests/test_v10_protocol.py tests/test_v10_env_contract.py tests/test_v10_teacher_yaw.py -q
```

结果48 passed、1个上游torch.jit.script弃用warning，21.59 s；git diff --check通过。trace自有8项测试覆盖旧rows20不变、源数组clone不污染、helper混合路由复用、首终止冻结、紧凑定长结构、源float32保留、容量填满及超限、shape/finite/mask拒绝和padding隔离。CPU假环境走真实run_suite循环，4病例分别在1/2/3/4帧终止，保存action yaw与next yaw严格相差一步；charged16=effective10+padding6，首失败组停止且不产生正式cache。

仅测试CPU数组/假环境及冻结ETH CPU前向，没有新增物理/GPU/probe/训练，没有资源通过或能力改善结论。全项目最终回归、代码提交推送与logs/codex_changes.md统一追加由根代理负责。
