# A2 v1.0 实施报告（工程完成、行为验证待执行）

本次按v1.1 SDS在main新增独立v10模块，未覆盖旧SDS，未提交原有无关修改。规范SHA：eb95905ed24bf0674f94a1b02dab02256ee11e40a20ed1cd96eb24daeb1b49a0。Q1/Q2/Q3独立裁定见reviews/v10_execution_rulings.md。

## 功能与训练逻辑

同一个学生读取45维本体、5×45历史、3×40×30真实高程图、3维路线误差。平地/独立平台由固定盲走教师标注，楼梯由指定S2教师标注；第二个合格登顶帧结束后，下一动作才切盲走教师。教师不接管学生、不写入未执行动作历史。部署只导出学生。

第一段DAgger：D1先整合宽平台和平地/楼梯；D2加入0.8m平台；D3加入完整登顶走停路线。六任务reservoir各32768条，CPU FP32；每逻辑批3072，每任务512，其中当前/历史各256，六任务等权动作MSE。每轮48控制步，1024环境20次Adam，512为10次，微批<=256；LR固定3e-5，梯度clip1。DAgger上限D1 24576000/D2 49152000/D3 73728000样本，不使用奖励更新Actor。

第二段PPO仅在DAgger独立终验通过后开始：冻结合格学生为T_keep，独立seed42初始化Critic且输出层清零。P0前4915200样本只预热Critic、std固定0.10；P1联合训练，std投影[0.05,0.30]。48步rollout、5epoch×4逻辑批、gamma0.99/lambda0.95、clip0.2、entropy0.003、keep系数1、LR3e-5固定。keep目标是合格DAgger学生，不是旧教师，旧样本不进入PPO回报。PPO总上限49152000，全部学习上限196608000。

六角色比例20/10/15/15/10/30%，控制0.02s、物理0.005s×4、地图每5控制步刷新、巡航0.3/0.5/0.6m/s。停车100控制间隔减速（或突停）、零速HOLD1000间隔、F_STOP重启5秒，P_STOP/楼梯停车重启5米。所有参数来自SDS，不增加噪声/随机化/延迟/外力或S3。

## 奖励与物理差异

新PPO奖励为r=0.02*sum(w_i*f_i)+0.4*dx*I_cruise+2*I_top+8*I_success-10*I_failure。连续项仅乘一次dt，位移/事件不再乘dt；同帧失败优先。详细20项原公式保持SDS第10章，不另改权重。旧零指令冒充稳定站立、关节越限填零接线被实际运动/真实四足安全支撑/真实软限位替代。C_raw为足力向量范数>5N；C_reward为当前或上一raw；safe_support是连续真实向上支撑+同高连通面内缩。新腾空计时按C_reward，明确与旧raw可能差0.02s。测量面与跌落路线参考面分离。

G/U primitive共用碰撞和地图，保留0.01m逐级踏面及14m平台，不用伪validity代替深落差。当前名义动作配置无raw clamp（clip=None），保持实际现有行为；动作目标仍q0+0.25*action，未隐改PD/物理精度。

## 门禁、费用和恢复

每2457600学习样本保存；DAgger各课程40/80/100%停点完成当前rollout的bootstrap/GAE/update后退出GPU，冻结同SHA依次A/B；均通过才晋级。PPO总25/50/75/100%检查，最多两次保持失败回退且不返预算。最终每阶段仅一次独立终验，不自动换新集合。

D1/D2/D3分别154/226/334组；A/B每组20例分别>=18，终验100例分别>=95，并检查相对教师/DAgger保持、完整窗口及配对覆盖。实际初始关节/出生位置不同，集合身份分离；失败/缺窗不删分母。资源探测1024优先、512回退，两档均完整DAgger/PPO栈5预热+20更新、填满replay、保存/恢复；整卡<=7680MiB、任务<=7168MiB、主机available>=4GiB。两栈同N均通过才正式使用，不在运行中换N。

零缓存评估上界1685390840，预计缓存节约517384320，学习和probe费用独立列账。评估每组20/100实际病例，首次终止冻结，padding零动作、全批物理费用照计。Q1外账分别报告charged/executed/uncertain/updated，预留不确定不退款。Q3事件链先落盘结果/事件/证书再更新controller，终验开封和回退次数不能回退。已验checkpoint原字节+sidecar交付。真实GPU子进程共享互斥锁，parent崩溃也不释放活跃child的租约。

## 现有验证及限制

- 新版统一CPU套件最终结果另见独立审核报告和修改日志；不得将工程通过当行为通过。
- 实际CUDA工程416控制样本：环境128+六角色运行时smoke288；D1/D3各构造0步确认机器人物理SHA一致、地形SHA不同。实际GPU检查包含三高度跌落、同帧观测、首次终止冻结及padding零动作。
- 独立审查已修复分母重复、缺窗绕过、NaN放行、端点时长、F_STOP距离混淆、漂移起点、固定教师恢复、地形污染物理SHA及Critic CUDA RNG等实现缺陷。
- 裸pytest有324项收集错误（历史logs重复测试及旧bootstrap冲突，首次还包含未完工新调度）；tests目录完整命令有109项旧版本bootstrap收集错误，保留/tmp/a2_v10_full_pytest.log、/tmp/a2_v10_tests_directory.log。不改旧测试，不声称全项目测试通过。
- 完整双栈probe、教师准入、正式训练、行为终验和导出尚未完成。当前无效果改善对比数据。

## 执行启动记录

- 独立工程报告通过，冻结源码指纹7f15a529b47f0b9a5512109a9050ad25a5b00c82bb294deaefd0edf0b7cb71a7；最终统一62 passed、4显式GPU skip。
- 实现提交4b80853ab7875970efbd6953e3223314dff094c8已成功push至origin/main，并用ls-remote逐字核对。初次HTTPS缺默认credential helper，使用本机已有gh登录凭据助手后成功；未改远端URL、未读取/输出明文token。
- 自动流程已启动：logs/a2_v10/formal_20261126_approved；当前正在资源探测。工程准入logs/a2_v10/engineering_admission_20261126.json，controller_state.json记录运行/停止状态。
- 启动命令：PYTHONUNBUFFERED=1 MUJOCO_GL=egl /home/xxs/anaconda3/envs/a2/bin/python scripts/train_a2_v10.py --run-dir logs/a2_v10/formal_20261126_approved --engineering-admission logs/a2_v10/engineering_admission_20261126.json --device cuda:0。
- N选择、峰值、教师逐组结果与正式学习样本以对应运行结果为准；此记录不声称正式学习已经开始。

## Q4资源失败后的修复进展（取代“当前资源探测中”的状态）

首次自动流程已严格停止：1024/512 DAgger均因available<4GiB未通过，实际probe累计737280，updated663552、uncertain0；无PPO probe、教师准入或正式学习。512虽完成25轮及实际保存恢复，最低available2883108864，不能判通过。旧4b80853工程实现还发现worker保存分支未定义变量遗漏，原审核结论标为发现遗漏、待关联集成重验，不删除原历史。

Q4仅修内存所有权/临时副本/8MiB流式哈希和保存状态引用，不改SDS训练配方。完整196608 FP32 CPU测试已实际通过（8 passed69.18s，0物理）：3100924643字节checkpoint完整SHA/CRC、释放旧库、真实加载、完整内容/metadata/int64/model/Adam/normalizer/RNG及下一批等价。RSS满库约4.136GB→释放约1.035GB→加载约4.153GB；这些是CPU测试证据，不是重probe通过。真实worker CPU集成覆盖首次/恢复后保存及阶段末去重。

全局探测账本已从原两份账本逐行核SHA、幂等迁移737280，跨新run不清零，未创建新额度授权。原execution_plan未明确全局B_probe，3833856只是单项上限推导的工作量需求；拟议累计4571136等待用户明确预算授权。新增物理采样停止，当前继续CPU等价和独立审核。最新审核与提交结果将另追加。

## Q4工程交付结果

- 新源码4005639f5656c64a81fbfa4499216b35f14c8b9eef2442c128f45a974d44b6a9；统一测试83 passed、5显式skip、1上游warning，81.97s。完整3GB项已单独实际通过，此处不重复分配；其余4项GPU跳过不当通过。
- 独立复审41 passed、1完整3GB skip，报告在原文后追加并明确首轮worker保存遗漏；追加后SHA a82a34377bd3825fd2168bbe4dfbddf4847aabba137b2ad06a626827a9c97166。新工程身份仅允许后续合规探测，旧身份已因代码/报告变化失效，不沿用。
- 首次资源失败的controller与selection完整证据存reviews/v10_first_probe_failure_20261126.json。当前没有任何正式学习checkpoint或教师准入结果，不存在可交付的合格学生模型。
- 已实现全局probe账本、所有入口缺授权拒绝、跨run不清零。授权总额尚待规划线程获取用户明确答复；未创建生产预算授权，Q4新增物理为0。不声称资源峰值已改善或新N已选定。
