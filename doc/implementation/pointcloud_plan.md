# SDS v1.2 工程实施记录

2026-11-10；基线 8ec99c785eae2ee4f4c8de69bfc9882ceda5173f，a2-pointcloud，起始工作区干净。

## 所有权与接口

- perception agent：perception.py / test_pointcloud_perception.py。
- model agent：model.py、algorithm.py、checkpoint.py、export.py / test_pointcloud_model.py。
- geometry agent：geometry.py、rewards.py、terminations.py、curriculum.py、evaluation.py / test_pointcloud_geometry.py。
- 主 agent：其余集成、注册、CLI、文档和唯一修改日志写入者。
- 路径均在 src/tasks/pointcloud/ 与 tests/ 下。不得修改他人所有权文件；接口修订经主 agent 协调。
- Actor 只接收 obs_history[B,5,45]、points[B,5,70,3]、mask[B,5,70]、ages[B,5]；特权标签与critic独立。
- 地形首阶前缘x=0；世界转换只允许平移与yaw；结束优先failure>success>timeout。

## 顺序与验收

1. 阅读SDS/代码/论文、冻结接口（进行中）。
2. 并行实现感知、网络优化、几何奖励课程及单元测试。
3. 主agent整合环境生命周期、训练/评估/压测/导出入口。
4. 本地仅静态、合成数据、有限无学习仿真验证；严禁任何训练。
5. 独立agent依据实际实现与失败模式审核，交回修复并复审。
6. 更新SDS状态、服务器操作与外部权重清单，汇报待提交范围。

服务器目标保持8096环境、rollout24、epochs5、minibatches64、4000迭代；保存25、验证100。训练与8096压测均由用户执行。

## 集成进度

- 感知显式两传感器扫描，机器人碰撞group3参与遮挡；静态四位置验证已执行。中段后雷达稀疏（4个ROI回波）记录为盲区，不改变外参。
- ETH真实权重初始化动作误差0；合成优化器单步、恢复、full/blind ONNX已由实施者检查，主复验中。
- 环境生命周期独占委派geometry agent（新增env_cfg.py/environment.py及环境测试）；其余原文件所有权不变。
- 主新增config.py/runner.py/validation.py和训练入口；model agent额外独占check/export/resource_test/evaluate脚本。
- 本地仅合成测试、静态射线及少量有限物理步；没有执行训练或压测。
