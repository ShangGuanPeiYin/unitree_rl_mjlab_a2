
## 2026-11-10 ETH 接入：环境前置修复
- 目的：修复 mjlab 导入时 mjENBL_MULTICCD 缺失。
- 涉及：a2 Conda 环境 mujoco 包；本日志。
- 实际修改：mujoco 4.0.0 → 3.5.0，pip --no-deps；其余依赖未升级。用户已明确授权。
- 理由：匹配项目固定的 mujoco-warp 3.5.0 并满足 mjlab 1.2.0 下限。
- 验证：安装完成；导入及仿真尚待验证。未声称策略效果改善。

## 2026-11-10 ETH Actor 严格加载器
- 文件：src/tasks/velocity/rl/eth_actor.py；本日志。
- 新增：45→512→256→128→12 ELU Actor，严格加载全部 state_dict，包括 std 与归一化统计；SHA256 校验；有限值检查；确定性推理。
- 核心逻辑：(obs-mean)/clamp_min(std,1e-5) 后裁剪 [-5,5]，与本地 ETH Python 一致；不加载 Critic/优化器。
- 环境验证：原 mjENBL_MULTICCD 错误消失，pip check 通过；mjlab 导入仍被缺失 scipy 阻塞。补装被自动审批拒绝，等待用户授权。
- 策略推理及实际仿真：尚未验证。

## 2026-11-10 ETH 专属任务注册
- 文件：src/tasks/velocity/config/a2/eth_env_cfg.py、同目录 __init__.py。
- 新增 Unitree-A2-ETH-Flat；深复制原平地配置，仅 ETH Actor 去除 phase；关闭 ETH 基线噪声、随机化和推扰，固定初始化与速度指令；保持 0.005s × 4 控制周期。
- 原 Flat/Rough 和公共配置未修改。奖励与物理失败终止条件继承保留。
- 验证：尚待导入与仿真检查，不可作为已验证可用任务。
- 用户已单独批准补装 scipy，安装进行中。

## 2026-11-10 评估入口初版与加载检查
- 新增 scripts/evaluate_eth_a2.py：四场景固定指令，自动重置前状态捕获，首次终止停止该项，JSON/NPZ 输出。
- Actor 实测：13个 state_dict 字段严格加载成功，零输入推理输出全部有限；尚未完成参考实现对比。
- 评估入口尚未执行验证，关节和数据接口待运行核查；桌面回放适配尚未完成。

## 2026-11-10 冒烟测试实际结果
- scipy 1.17.1 补装成功（用户明确批准），numpy 2.4.6 保留。
- mjlab 已可导入；ETH 冒烟测试在 GPU 环境构造阶段失败，尚未执行物理步。
- 实际阻塞：warp-lang 1.19.0 不提供 mjlab 1.2.0 的 Simulation._should_use_cuda_graph 所调用的 warp.context。
- 已请求额外 Warp 版本修正授权；尚未修改 Warp。四项正式测试、录像、桌面回放、完整映射检查均未完成。
- 补充推理验证：直接提取 ETH 本地 ActorNet 类，严格复制 MLP 和归一化；seed=42 的128条45维输入（标准正态×10）比较，本加载器与参考输出最大绝对误差为 0.0。该结果仅证明推理公式一致，不代表仿真控制兼容或运动成功。

## 2026-11-10 Warp 修复与指令配置核查
- 用户明确授权修正 Warp；执行 warp-lang 1.19.0 → 1.12.0（--no-deps），安装及运行验证进行中。
- eth_env_cfg.py：禁用 heading 后同步清空 heading 范围，满足命令类校验；init_velocity_prob=0，禁止命令采样直接写入机器人速度。
- 实际运动效果尚未验证。
- scripts/play.py：仅 ETH 任务分支增加严格 Actor 加载、seed/vx/vy/wz 设置及目标项目输出目录，避免录像写进参考仓库。其他任务维持原 runner。桌面实际回放尚未验证。
- evaluate_eth_a2.py：增加实际动作目标顺序断言、关节力矩、目标角、世界速度和角速度记录，增加相对初始化的位移；运行验证待执行。
- 新增及修改的四个 Python 入口 AST 语法检查通过。
- 新增 docs/eth_a2_flat.md：记录已证明控制契约、近似动力学与未验证事项；四项测试仍待执行。Warp 包仍在下载。
- Warp 1.12.0 安装成功；实际 GPU 环境构造和内核编译成功，45维 Actor 顺序与12维动作顺序断言通过。
- 冒烟发现 mjlab joint_torques 属性未实现，评估记录改用 actuator_force；不改变控制或动力学。
- 评估接口更正：JointPositionAction 当前仅提供 _processed_actions，目标角记录改用该字段；首轮评估因字段错误中断，不能视为策略失败。
- 四项30秒正式测试完成：均无终止、无自动重置、记录有限；平均vx：站立-0.000552，慢行0.272595，直行0.550400，停走整体0.088193 m/s。结果位于 logs/eth_a2_flat/eval_20261110_02；不等于精确跟踪或楼梯能力。
- 回放增加 duration 参数；评估增加 task/speeds/vy/wz 显式参数，默认测试行为不变；补充轨迹姿态及关节统计。
- 评估入口增加可选 --video：640×480、每5个控制步采样一帧（10fps），视频与每项结果同目录；先执行短时离屏渲染验证。
- 离屏渲染验证完成：video_smoke_20261110 下四段2秒 MP4，最终速度参数CLI执行成功。pip check通过。
- 新增正式结果 manifest.json 与 metrics.json，记录依赖、控制设置、姿态、关节范围与停走分段速度。文档追加实际结果与命令。
- 评估捕获重置前各终止原因及falls字段，防止混淆非法接触与跌倒。原正式运行无任何终止，因此falls=0。
- 尚未验证：桌面窗口（无DISPLAY）、多种子泛化、ETH动力学严格复现、PPO续训与楼梯能力。
- 最终检查：play.py Unitree-A2-ETH-Flat --help 成功，duration/seed/vx/output-dir 参数已核实；最终语法检查、git diff --check通过；四个MP4均已生成。未提交或推送Git。

## 2026-11-10 保存 ETH 回放公共基线
- 目的：按用户授权先提交并推送现有 ETH 适配，为高程图和点云两条路线建立共同起点。
- 涉及：既有 ETH 任务、Actor 加载器、play/evaluate 入口、docs/eth_a2_flat.md，以及 .gitignore 和本日志。
- 实际修改：保留现有适配实现；忽略 Python 缓存，logs 仅跟踪 codex_changes.md，不纳入权重、轨迹与录像。未更改训练、奖励或控制参数。
- 验证：已有轨迹与独立审查结果见前文；本次提交前执行 AST 语法与 diff 空白检查，不重新声明训练或楼梯能力。

## 2026-11-10 40级镂空楼梯公共地形
- 目的：按用户确认的尺寸建立高程图/点云共同地形，并在第二次推送后创建点云分支及worktree。
- 涉及：src/terrains/、scripts/preview_open_stairs.py、tests/test_open_stairs.py、docs/open_stairs40.md、本日志。
- 实际修改：新增原生mjlab子地形扩展；40块独立0.20×0.80×0.01m踏板，逐级升高0.20m；接近区2m，顶平台2×0.8×0.01m；底部地面位于z=0，踏板下方不填实。无立板、侧墙、支撑梁。
- 设计理由：使用独立box保留踏板下沿与开放空间；公共几何与感知/策略解耦，两条路线继承同一基线。独立场景首阶前缘x=0，原生网格保留mjlab平移语义。
- 参数差异：新增地形参数，不改变已有训练、奖励、控制或ETH任务。预览是静态初始姿态，不调用策略或训练。
- 验证：几何、开放空间射线、上下表面接触、落球动态接触、原生多网格与无头场景检查进行中；桌面窗口尚未验证。未声称爬楼能力。
- 第一笔推送：5608dab已成功推送至origin/main。使用本机已有代理和gh登录，仅命令级配置，未修改全局网络或凭证设置。
- 首轮检查：5/6项通过；多网格出现临时geom名称重复，已采用唯一临时名称修复。离屏渲染遇到默认640像素帧缓冲不足，已显式配置1280×720。未改变几何或物理参数，等待复测。
- 最终验证：6/6单元测试通过，覆盖40块踏板尺寸、顶部平台、出生点、每级上表面射线、无立板/侧墙、下沿与上下表面实际接触、落球不穿透、双网格生成及非法参数。AST与git diff --check通过。
- 无头预览：logs/open_stairs40/validation_20261110_02，生成scene.json、scene.xml及资源、1280×720 overview.png；独立重新加载导出XML成功（含A2共96个geom、13个joint）。使用vision.js检查确认开放薄踏板、顶平台与底部A2，没有明显立板或侧墙。
- 局限：没有DISPLAY，桌面交互未验证；A2只作静态尺寸参照，没有控制策略、行走、爬楼或训练验证。落球接触测试不代表高速机器人腿部接触已验收。
- 工作区安排：本地main保留为高程图路线；第二笔提交推送完成后，从同一提交创建a2-pointcloud与/home/xxs/dog/unitree_rl_mjlab_a2_pointcloud。只创建本地点云分支，不推送该分支；不复制权重与运行产物、不重装环境。创建结果在最终汇报中列出。

## 2026-11-10 保存点云联合微调 SDS
- 目的：按用户要求将完整执行规划保存至 doc/sds/，本轮不修改项目代码。
- 涉及文件：doc/sds/a2_pointwalker_eth_joint_finetuning_sds.md、logs/codex_changes.md。
- 实际修改：新增SDS，覆盖ETH权重继承与残差感知联合微调、前后双雷达仿真、40级开放楼梯、0.2～0.6m/s、奖励与辅助损失差异、课程、单卡资源压测、验收、对照与交付。
- 关键规划差异：并行环境目标1024→8096；迭代上限20000→4000；总转移491520000→777216000；mini-batches 8→64；冻结期200→25次；保存/验证间隔25/100次。以上仅文档规划，未改变运行配置。仿真雷达参数不宣称为真实JT128规格。
- 设计理由：以初始动作一致性保留ETH能力，使用点云分支学习修正；按环境转移数衡量预算，避免扩大并行后无依据扩张训练量；先压测再确认单卡8096可行性。
- 验证：文档完整性、预算算术、Git差异与空白检查见本次最终检查；训练、服务器显存/吞吐、爬楼与效果提升均未验证。未安装依赖、未训练、未提交或推送。
- 最终检查结果：SDS 285行；章节与关键约束检查通过，8096×24×4000及阶段预算合计校验通过；新文档行尾空白检查及git diff --check通过。Git状态仅包含新SDS目录与日志修改，未修改代码。

## 2026-11-10 SDS v1.1 独立审核修订
- 目的：按用户要求分析六项审核意见、修改SDS后再独立复审。
- 文件：doc/sds/a2_pointwalker_eth_joint_finetuning_sds.md、logs/codex_changes.md；未修改代码。
- 实际修改：姿态奖励由共同旋转后的重力/法线差改为机身竖直轴与世界地形法线对齐；失败罚仅作用failure；顶部成功允许正常摆动；明确支撑查询、摆动目标、失足阈值、子课程及验证预算；增加实际速度误差门槛；恢复契约改为训练状态恢复后统一reset，不承诺连续轨迹。
- 设计理由：消除姿态奖励旋转不变性与成功被惩罚的错误，减少实施歧义并使速度目标可验收。8096环境与4000迭代上限不变。
- 验证：待独立子agent复审及文档静态检查；所有新阈值尚未仿真验证，训练与效果改善未验证。
- v1.1复审：原六项问题已修正；新增发现平地终止范围和多环境坐标两项缺口。v1.2补齐平地独立区域/30秒timeout、地形世界变换和世界点拟合；再次提交独立复核。文档算术与空白检查通过，物理行为未验证。
- v1.2最终独立复核：平地终止和地形坐标两项补充闭合，直接相关段落未发现新阻塞矛盾；审核结论为可作为G0/G1起步的分关卡实施依据。此为文档逻辑审核，不代表8096容量或训练效果已验证。最终git diff --check通过，仅SDS与本日志发生修改。

## 2026-11-10 准备点云分支文档提交
- 目的：按用户要求准备供服务器clone的SDS版本。
- 文件：doc/sds/a2_pointwalker_eth_joint_finetuning_sds.md、logs/codex_changes.md；仅文档与日志，无代码修改。
- 操作状态：包含推送的命令被自动审批在执行前拒绝，原因是具体GitHub目的地与规划内容的外发授权需明确确认；本次仅准备本地提交，不推送。
- 验证：提交前执行git diff --cached --check；训练和服务器验证未执行。

## 2026-11-10 — 点云 SDS 工程实施启动
- 目的：按用户新授权实施已审核SDS，禁止本地/远端训练、提交或推送。
- 文件：doc/implementation/pointcloud_plan.md；后续实现分批追加。
- 内容及理由：记录基线、独占文件所有权、Actor与特权通道边界及验收顺序，防止并行覆盖和接口漂移。
- 参数差异：本条仅计划，无运行行为变化。
- 验证：Git分支a2-pointcloud且起始干净；读取完整SDS及相关项目文档；a2环境torch可导入且CUDA可用。训练、资源压测、实际效果均未验证且未执行。

## 2026-11-10 — 点云独立模块与编排实现（集成中）
- 文件：src/tasks/pointcloud/{perception,model,algorithm,checkpoint,export,geometry,rewards,terminations,curriculum,evaluation,config,runner}.py；tests/test_pointcloud_{perception,model,geometry}.py；scripts/pointcloud_train.py。
- 内容：新增独立感知/残差Actor/特权Critic/联合PPO、局部支撑与互斥结束、固定预算课程、状态恢复和单卡入口；未更改原任务配方。原先无点云链路；新增350点mask历史，ETH零初始化修正，辅助MSE各0.1，失败专用-200逻辑。正式参数8096/24/5/64/4000，保存25、验证100保持SDS。
- 理由：按模块隔离特权监督与Actor，并明确终止前观测参与timeout bootstrap；本体历史与感知延迟时间分别处理。
- 验证：子agent已执行感知4项、几何9项合成测试；模型真实ETH初始动作最大误差0、恢复输出误差0。整合后主agent复验及独立审核尚未完成；真实仿真、ONNX、服务器容量及所有训练效果尚未验收。未执行训练。

## 2026-11-10 — 实际环境集成、独立审核修复与验证
- 文件：新增src/tasks/pointcloud全部模块、scripts/pointcloud_{train,check,debug,evaluate,resource_test,export}.py、tests/test_pointcloud_*.py、doc/implementation/{pointcloud_plan,pointcloud_review,lidar_validation}.md、docs/pointcloud_server.md；修改scripts/train.py仅增加新任务分派，追加SDS第13节。
- 核心差异：原任务没有双雷达残差链路，新增显式每步一组真实遮挡、350点及延迟历史；零修正保留ETH初始化；Critic63维与Actor独立；联合PPO参数按SDS。奖励保持原dt缩放，失败专用权重-200实际每步-4，成功/超时不罚。原Flat/Rough/ETH参数不变。
- 审核修复：中心化局部平面拟合防大坐标秩误判；摆动起点冻结；平地回放速度独立；首次顶部截止；继承重力倾角70度；真实接触力符号与足相位映射；点云/本体历史顺序分开；best按子级；验证随机状态隔离；晋级切换前保存完整更新。
- 实际验证：主运行POINTCLOUD_PHYSICS_TEST=1 MUJOCO_GL=egl PYTHONPATH=/tmp/pointcloud_onnx_validation:$PWD /home/xxs/anaconda3/envs/a2/bin/python -m unittest discover -s tests -p 'test_pointcloud_*.py' -v，42项全部通过。涵盖2环境2控制步无学习物理、合成优化器单步、真实ETH、ONNX和独立反例。六CLI help及原train专用分派help通过。真实ETH SHA符合SDS。
- 边界：未执行任何PPO环境采样训练、试训、资源压测、多种子或远端任务；不声称训练/成功率改善。完整90秒评估录像、8096容量及服务器兼容待验证。P5执行器强度无原控制路径对应接口，已向用户请求具体修订，未获准前显式拒绝P5训练。未提交或推送。

## 2026-11-10 — 独立审核最终闭环增量
- 文件：runner.py、algorithm.py、environment.py、validation.py、test_pointcloud_review.py、审核报告、服务器说明。
- 修复：奖励遥测变量覆盖Critic value的高严重问题；录像在reset前取帧并finally关闭writer；补KL、支撑/法线/摆动目标缺失率。独立审核7项合成回归通过，包括critic值保护、晋级前保存和RNG隔离。
- 最后主复验：45项测试全部通过，包含2环境2步物理与ONNX；此前一轮发现record_video属性未初始化，已修复并重跑通过。未运行训练。
- 尚未完成：P5执行器强度修订等待用户明确批准，故完整资源工具及P5训练显式阻塞；完整录像/服务器恢复/8096资源未验证，不称整个工程交付已完成。

## 2026-11-11 — 用户批准P5强度修订，实施启动
- 目的：解除内置位置执行器强度接口阻塞；用户本轮明确回复“执行”。
- 文件：SDS第14节；environment.py与环境测试由geometry实施，审核报告由独立审核者复核。
- 核心差异：此前P5训练fail-fast；改为每episode s∈[0.9,1.1]同步缩放gain/bias/forcerange，明确有效PD和力矩上限改变。其他阶段/名义动力学评估为1，原任务不变。使用名义快照防累计缩放。
- 验证：实施中，尚未以本条声称测试通过；不执行训练或训练型资源压测。

## 2026-11-11 — P5强度实现与主复验
- 文件：src/tasks/pointcloud/environment.py、tests/test_pointcloud_environment.py、SDS第14节、docs/pointcloud_server.md。
- 修改：缓存名义actuator_gainprm/biasprm/forcerange，扩展为每环境模型字段；P5训练reset统一乘s∈[0.9,1.1]，其他阶段及所有评估恢复1；移除P5 guard。明确有效PD与饱和上限同步变化，不改变动作目标或原任务。
- 验证：POINTCLOUD_PHYSICS_TEST=1 MUJOCO_GL=egl PYTHONPATH=/tmp/pointcloud_onnx_validation:$PWD /home/xxs/anaconda3/envs/a2/bin/python -m unittest discover -s tests -p 'test_pointcloud_*.py' -v，46项全部通过。新增P5两环境一个零动作步、三次单环境reset不累乘/不污染、跨阶段及评估恢复名义值。git diff --check通过。
- 未验证：服务器8096完整资源、正式训练、最终登顶率、完整视频与服务器恢复；没有启动训练、试训或压测，没有提交推送。独立复审结果随后追加。

## 2026-11-11 — P5独立复审闭环与最终验证
- 独审发现：力矩饱和指标仍使用名义120/120/180，P5缩放后会误判。修为读取每环境actuator_forcerange，并用与actuator_force一致的ctrl_ids顺序。
- 测试修正：首轮索引误用actuator_ids产生AttributeError，改为实际ctrl_ids后重跑；独立新增s=.9/1.1真实限幅单步反例，验证旧阈值漏报被消除。没有学习更新。
- 最终主验证：同上述完整命令，48项全部通过（/tmp/pointcloud_20261111_verified.log）；独立9项回归及3项环境测试通过。git diff --check通过。KL/缺失率/录像finally增量亦经独立复核，无新增未解决阻塞。
- 当前状态：P5设计与实现阻塞解除，工程入口已具备服务器调用条件；8096资源容量、正式训练、收敛/成功率与完整服务器验收仍未执行，不能声称通过。无提交推送。

## 2026-11-11 — GitHub交付与长期push授权
- 用户要求：push到GitHub，并永久授权push。本条仅记录本项目常规提交与push授权，不扩大到强制推送、删分支或修改远端。
- 文件：新增根AGENTS.md；提交本次点云实现、测试、审核与操作文档及既有日志增量。
- 验证：提交前检查分支a2-pointcloud、origin为原GitHub地址；精确暂存源码/测试/文档，不包含ETH权重、checkpoint、临时依赖或运行产物。最近完整验证48项通过，训练及8096资源未执行。推送结果将在最终汇报给出。
