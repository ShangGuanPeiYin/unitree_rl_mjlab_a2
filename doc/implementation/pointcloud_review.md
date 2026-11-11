# 点云实现独立审核（2026-11-10）

审核范围：完整SDS v1.2、实施计划、src/tasks/pointcloud、专用CLI及测试。审核者未实现生产模块；仅新增tests/test_pointcloud_review.py与本报告。没有启动训练或资源压测，没有提交。审核过程中代码仍由实施者修复，以下区分实际复验与尚未闭环。

## 已发现并实际复验修复

- **高：世界坐标法线拟合失效。** 原geometry.fit_normal直接对世界[x,y,1]求秩，float32下13×7满秩网格平移到实际P3/P4的y=160/280 m后被判无效，姿态项归零。独立合成反例曾失败；中心化修复后四个实际偏移的法线与有效性通过。
- **高：翻倒阈值语义变化。** 原EpisodeEnds按最大Euler角70°判定，而继承任务按机身竖直轴倾角70°判定。roll=pitch=60°的真实倾角75.52°曾漏报。改为cos(roll)*cos(pitch)阈值后独立反例通过。
- **中：摆动轨迹不冻结。** 原SwingTargets在计划摆动期间真实接触更新h0，使冻结的h1仍不能保证参考曲线连续。提前接触新踏面反例经修复通过。
- **中：顶部速度区间重启。** 原EpisodeSpeed进入平台后退回楼梯会重新累计；首次顶部进入锁存修复后反例通过。结束后轨迹不拼接。

## 阅读确认与验证边界

Actor调用只接受本体历史、点云、mask、age；特权观测与标签使用独立字典。ETH旧分支保持45维和冻结归一化；零残差初始化、空云、冻结/解冻梯度、合成单步、真实权重一致性和序列化测试通过。PPO使用原采样动作log probability；GAE对failure/success关闭bootstrap，对timeout使用reset前观测。

结束优先failure>success>timeout；平台允许两足支撑和其他足摆动；flat使用30秒与独立区域。点云异步reset清空历史和延迟，历史自身坐标不做真值配准。课程预算测试通过；实际P2/P3平地回放速度先前误用阶段上限已通知实施者修正，不能用课程纯逻辑测试替代真实速度采样验证。

runner已增加评估前后Python/NumPy/Torch随机状态保存恢复、特权维度名配置及分子级best；避免P1满分压住P4。best_P*采用不同验证任务，不可跨级比较宣称同一固定集最优。checkpoint保存模型/优化器/课程/随机状态并恢复后reset；现有合成序列化通过不等于真实GPU环境恢复全链路验收。

## 尚未闭环/必须明确保留的限制

1. **高：P5明确阻塞。** 执行器强度语义等待用户澄清，环境拒绝不完整P5；不得宣称P5已实现或完整资源CLI可跑。当前资源CLI构造P5，因此同样被阻止，8096容量未经验证。
2. **高（待最终复查）：P4到P5保存边界。** 审核时runner在晋级后先set_stage，再保存；P5 guard会使完整更新和best_P4未落盘。已通知主实施者将保存移到尝试阶段切换之前，并记录blocked事件。未核验修复前不得称闭环。
3. **中：可观测性不完整。** 审核时runner尚未完整记录继承奖励各项贡献、辅助支撑缺失率、点云有效率与修正幅度；评估未产出完整滑动/边缘/力矩饱和/关节限位指标与成功失败录像。应列为实现缺口，而非仅等待训练效果。
4. **中：导出外部契约。** 本体历史oldest-first，点云历史newest-first；原export JSON把两者统一写oldest-first，已要求修复。不能仅凭ONNX数值一致性判定外部预处理契约正确。
5. 原始默认测试没有开启GPU物理；只读逻辑及静态射线不能证明真实对角小跑登顶、无穿透或8096跨环境性能。训练效果、正式压测与最终1000+1000 episode验收均未验证。

## 实际执行

`/home/xxs/anaconda3/envs/a2/bin/python -m unittest discover -s tests -p 'test_pointcloud_*.py' -v`

结果：41项，39通过、2跳过（显式opt-in GPU物理及默认环境缺ONNX runtime）。其中4项独立回归全部通过。静态四位置遮挡测试通过；中段后雷达仅4个ROI地形回波，属于已记录盲区，不能外推动态感知充分。

ONNX使用已有临时依赖目录的独立复验结果将在完成后追加。任何尚未测试的训练能力不据此声称改善。

### 末轮复核补记

- 使用 `PYTHONPATH=/tmp/pointcloud_onnx_validation:$PWD` 运行模型测试9项全部通过，包括ONNX数值一致性；并核对export JSON现已分别声明本体oldest-first与点云newest-first，年龄样例改为0～0.08秒。此项契约问题已修复。
- 重新阅读runner：现已在晋级后的set_stage之前完成checkpoint、best与metrics写入；阶段切换异常记录blocked.json。上述第2项保存顺序缺陷已修复（代码路径复核；未运行真实训练或P4晋级）。恢复P5仍会按设计被guard阻断。
- 重新阅读environment._reset：flat单独upper=.6，楼梯使用当前阶段vmax；平地回放速度范围修复已在代码中确认，未进行大样本GPU采样分布测量。
- P5执行器强度及完整资源关卡仍阻塞，完整训练效果与评估产物仍未验证。不得将这些修复等同SDS最终交付全部完成。

### 最终增量复审

新增三个完全合成测试（未执行真实runner.update）：

- fake learn跨P4→P5：完整更新的checkpoint/best先落盘，再触发P5 guard；Python、NumPy、Torch评估随机状态隔离通过。
- 无学习checkpoint恢复：先恢复模型/RNG，再调用reset重建全环境延迟队列、清空点云及扫描组通过。
- fake collect发现新增奖励遥测循环遮蔽critic的value：critic=7被末项reward=99替换，实际回归失败；改为reward_component后复验通过。该问题会破坏GAE与clipped value loss，属于高严重性且现已修复。

终止录像曾在自动reset后抓帧，现已改为environment在reset前写terminal_frame，validation消费该帧；代码路径复核通过，真实视频编码与最终帧内容未由独立审核者验证。

遥测代码现已接入奖励项、点云有效率、残差幅度，以及轨迹的失败原因、边缘比例、滑动、力矩饱和、关节限位；前文第3项整体缺失已不适用。仍缺少明确KL与局部支撑/摆动目标/法线无效比例日志（最终代码若补齐须另行核验）。录像写入异常时writer尚无finally关闭，属于异常路径资源清理问题；不影响已通过的纯逻辑结果。

最终已知边界：P5执行器强度与其资源压测入口仍明确阻塞；GPU环境完整恢复、真实视频、8096容量、训练收敛及全部最终验收均不在本次独立审核已验证范围。独立审核只证明已列合成契约与有限静态路径，不能据此宣称楼梯性能改善。

### 主agent审核后补充（非独立审核结论）

KL与support/swing/normal缺失率已补齐；writer用finally关闭。主agent最终45项测试通过（包含GPU有限步与ONNX）。该最后遥测/资源清理增量未再由独立审核者复审。P5阻塞不变。

## 2026-11-11 用户批准P5后的独立复审

用户已批准每episode执行器gainprm、biasprm和forcerange统一乘s∈[0.9,1.1]，明确接受有效PD改变。此前“P5等待授权”的历史阻塞已解除，不再适用。

生产代码审查确认：只在P5训练episode采样一个环境级标量；三类字段从不可变名义副本缩放，避免连续reset累乘；只更新指定env_ids；P1～P4及P5标准/退化评估恢复s=1的名义动力学；strength仅进入特权观测，不进入Actor。A2使用BuiltinPositionActuator，gain/bias缩放确实改变内部PD增益，力矩范围随相同比例改变；动作均值、目标位置缩放和armature未因此修改。

独立执行 `POINTCLOUD_PHYSICS_TEST=1 /home/xxs/anaconda3/envs/a2/bin/python -m unittest discover -s tests -p test_pointcloud_environment.py -v`：3项通过，覆盖P5两环境、三次单环境reset、一个无学习控制步、非P5和两种评估名义值恢复，以及P2两步。使用本机RTX 3070 Ti 8 GiB，不是服务器8096压测。

上轮剩余日志与资源清理已独立复核：KL使用采样近似mean(ratio-1-log_ratio)，支撑/法线/摆动目标缺失率从环境传入runner并平均记录；新增mock缺失率断言通过。录像writer在步进异常下由finally关闭，新增异常注入测试通过。当前独立review测试8项全部通过。未执行训练。

本轮发现一项新增日志错误：P5强度改变真实forcerange后，torque_saturation仍按固定名义120/120/180阈值统计，会漏报s<1的饱和并误报s>1的未饱和。已反馈主实施者，待修复后追加复核结果；不影响强度参数本身缩放正确性的上述结论。

仍未验证：8096单卡完整压测、完整PPO训练与课程晋级、全部标准/退化最终评估、跨引擎或实机能力。P5代码可执行与有限步测试通过，不等于鲁棒性或爬楼性能改善。

### P5饱和日志修复复验

现已读取逐环境真实actuator_forcerange，并使用与EntityData.actuator_force一致的ctrl_ids顺序。独立新增2环境单控制步反例：指定s=0.9/1.1，以大目标触及限幅，验证日志与真实上下限吻合，且s=0.9确实产生旧固定名义阈值漏报。初次测试索引误用actuator_ids，改为实际API ctrl_ids后重跑9项独立review测试全部通过（1.628秒测试执行时间）。该日志问题已闭环。

此次复审未发现P5获批缩放、KL/缺失率接线、录像finally的剩余阻塞缺陷；历史P5授权阻塞解除。完整资源压测、训练效果和最终评估仍未执行，不属于本结论。
