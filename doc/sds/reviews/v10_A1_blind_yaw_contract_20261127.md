# A1统一blind yaw-only教师契约（2026-11-27）

## 批准与适用边界

用户已授权统一纠偏、交规划技术裁定。规划线程01a217c8-d825-7922-bd23-eb6ba2e37bfb于2026-11-27以A1批准以下精确公式的实现及CPU测试，物理/诊断/资源预算另行裁定。本契约关联并修订冻结SDS v1.1零vy/wz、无偏移回中的教师输入边界；原SDS文件及SHA eb95905ed24bf0674f94a1b02dab02256ee11e40a20ed1cd96eb24daeb1b49a0原样保留。

## 精确规则

输入仅学生同帧已有proprio[6]请求vx及route_error三维r。角度均为rad，转向rad/s，y单位m：

```
y = 0.2*r[0]                    # 可观测截断误差，范围[-0.8,+0.8]，不读取env原始y
psi = atan2(r[1],r[2])
psi_des = clip(-atan2(0.2*y,max(vx,0.1)),-0.15,+0.15)
e = atan2(sin(psi_des-psi),cos(psi_des-psi))
wz = clip(0.5*e,-0.15,+0.15)*clip(vx/0.3,0,1)
teacher_command = (vx,0,wz) if vx>0 else (0,0,0)
```

正y产生向负yaw方向纠偏。横向误差反馈系数0.2/s、目标yaw限幅0.15、角度反馈增益0.5/s、yaw-rate限幅0.15及0.1/0.3速度尺度为本次新增经验参数，并非已验证最优值。vy始终0，不同时引入横移控制。无滤波、积分、速率状态、RNG消耗或新增状态恢复负担；重复同输入得到同标签。

F/P及登顶后blind运动统一执行；不读hidden taskID、phase、platform geometry或privileged速度。vx>0的DECEL只随已请求速度连续减弱；STAND/HOLD请求vx=0则全零，不主动停车回中。RESTART按同正vx规则重启。楼梯S2输入、权重和路由不变。协议范围只覆盖原SDS非负前向请求，vx<0防御处理全零不代表新增倒行功能。

## 输入隔离与身份

仅blind教师私有clone proprio，在归一化前替换6:9。学生同帧proprio/history/map/route、环境请求与奖励目标不改，上一实际动作33:45保持真实学生执行值；blind内部命令不得进入学生历史或S2输入。Teacher仍只有原冻结ETH权重，不新增第三教师，网络权重SHA 8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722保持，但完整教师策略身份已变为权重加A1适配规则。新源码fingerprint与本契约共同标识，旧验收/缓存/资源证书不自动继承，不允许重写原证据code_sha。学生/优化器结构、损失/奖励、速度请求、课程、rollout48、N规则、checkpoint周期及学习额度都不改。

## ETH命令范围核查

model_2000.pt仅actor/critic/optimizer/iter2000/infos.env_state，无原训练cfg或命令上下界；normalizer命令vx/vy/wz均值约(.237525,.000245,-.001149)，std(.425382,.281530,.530494)，只能说明历史输入含非零横移/转向分布，不能推出精确上下界或保证响应能力。当前仓库通用训练配置vy/wz±1，A2 play配置±.5，eth_env_cfg为推理零命令；这些均不能证明该权重的原训练范围。0.15限幅是本次谨慎候选，效果待物理验证，不声称已改善。

## 验证要求与执行状态

CPU须覆盖纠偏正负方向、±pi wrap、yaw/速度限幅、零/负vx、减速趋零、确定性、clone无别名污染、同观测同标签、mixed blind/stairs和实际原ETH权重的适配输入parity。独立审核后才能称工程实现通过。尚未新增物理；完整教师准入依旧必须重新通过，不沿用旧平地200/200。
