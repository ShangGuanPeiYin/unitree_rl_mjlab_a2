# A1限定独立审查：spec compliance 与 quality

审查时间：2026-11-17T20:51:29.007545+08:00。Reviewer=/root/c1_reviewer（gpt-6.1-sol/high派工）。任务仅覆盖A1原生spec/geometry/budget/env的8个owner文件及4个测试；不涉及B核心或A2新worker/validate，未实现任何被审模块、未改生产或测试、未重跑已通过同SHA局部套件、未GPU/训练/Git写入。

## 双verdict与分级

- **Spec compliance：PASS_WITHIN_A1。** 在本报告绑定的8文件范围，符合SDS原生身份、追加D22/独立课程顺序、48000预算及不变计算路径的要求。
- **Code quality：PASS_WITHIN_A1。** 未发现需要A1生产修正的阻塞缺陷；没有新增分散地形常量或用最大物理ID推导最终任务，关键参数统一派生，原计算路径有明确保全证据。
- A1已确认缺陷：P0=0、P1=0、P2=0、P3=0；A1 blocking=0。以下“不能判定项”是跨模块/最终证据边界，不伪列成A1已实现缺陷。
- **整个v07项目：PENDING，不放行。** B/schema9/谨慎退级与恢复事务、A2worker/validate以及同freeze完整CPU/GPU尚未验收。本报告不是最终INDEPENDENT_ACCEPTANCE，不生成整体accepted=true。

## 精确审查对象

SDS SHA303ae40a5c5e3fe82f9f0df3aae8d153136e016439087c8db9701aeb17197aca；Qv07-001恢复初始化补充SHA1f84b142b70cc952eeca6e0a630f76d5d28f3d3b542cf2bcf5084d19be94a3cf。已读TASK_A_BRIEF、完整TASK_A_REPORT、全部1775行/85840字节审查包。包SHAfa03eefec81337053bc5b846e2c91caa664f04bfec943a0084a4d90363c0af4e与根指定一致；A_OWNER_SHA256逐项核对当前文件无漂移。

## Spec compliance逐项结论

|要求|独立审查结论与实际证据|
|---|---|
|原生身份|spec直接声明0.7.0 /0.7-d22-eval-regression-r1/schema9、旧architecture；末尾C1环境激活已删除。外部错误A2_V06_CONTINUATION的隔离子进程测试明确不导入v06.spec。|
|D22物理与最终任务分离|TERRAIN_ROWS末尾追加D22=tile22，PHYSICAL_TILE_COUNT23、FINAL_TILE_ID21显式分开；原0..21各TerrainSpec含原点/bounds完整asdict对拍。|
|独立课程顺序接口|CURRICULUM_TILE_ORDER=(1..17,22,18,19,20,21)，不含flat；near/easy消费归B，A只给单一顺序数据，没有以物理ID重排原几何。|
|D22几何|6级、h.20/d.22、宽.8/板厚.01/平台2；原点(0,176,0)、平台起点1.32/顶高1.20、身体线2.12均由TerrainSpec派生。实际ray顶面、query高度、安全bounds和接触tile索引有对应局部GREEN。|
|23数组与合法范围|terrain查询上界使用PHYSICAL_TILE_COUNT；env origins/N/h/d按完整table生成，geometry/map/contact/event数组动态派生；schedule允许22拒23。env _validate_scene的>len修为>=len，拒绝合法索引上界本身，符合23tile接线范围。|
|同一参数来源|造盒、解析query、events reward-height/timeout/platform support与map均经native TerrainSpec；timeout .3/.5/.6三路径锚点1828/1297/1164由参数测试覆盖。没有为D22另加偏置/epsilon。|
|预算与CLI合同|D25/D22/D20=6000/6000/9000，D21000，global48000；其余阶段不变，CLI预算是断言。总样本1179648000/物理4718592000/Adam960000由同预算派生。原费用迁移和formal48100账本归B，不在本8文件里假称已验。|
|保持的计算行为|spec的PPO配置、奖励配置除身份字段、资源/观测/动作尺度、出生/探索规则与C1相同；8文件未修改奖励/损失实现。env业务计算除合法上界检查外保持，终止公式/接触状态机没有放宽。完整v07 reward/PPO函数对拍仍需B及最终冻结验证。|
|路由|本地routing6项覆盖D25/D22/D20三速60例表、保持0/1各15、final L40/21共300表。它验证案例与reload身份，不是实际跑完这些物理case；B的evaluation改动不属于本A1质量签收范围。|

## 不变路径的独立静态核查

审查包直接基于before保存的C1源码，不使用当前HEAD作为未经核实基准。reviewer用纯AST读取比较：env_cfg/map/contacts/events/speed在去除文档与版本名后AST一致；terrain的tile_boxes、color、generator、index_geometry、build_terrain、表tensor派生、query_surface、platform_start/bounds、timeout_steps及tensorize_geometry计算一致。env差异为版本/身份/错误文案和唯一的geometry max>=len合法性修正；没有奖励、出生采样、control step或终止比较的计算改动。所有检查为只读解析，不执行测试或环境。

before清单的97个旧C1生产文件、28个旧测试及8项保全原件最终逐SHA重算一致；含唯一stop4817、原control/result/stop/terminal和旧独立报告。未追改旧冻结清单。before manifest SHA109fdb98eecf597604cbb66d58e5806798bf6116447da4a7045bd1dd5e3e2d56。

## 浮点边界审查

这是数学同值的不同舍入路径，不能混同证据等级：

1. float64单元把x_top=1.32、body_x=2.12用于理想边界/2.119反例，检查原support debounce和严格>=，没有生产epsilon变动。
2. 独立读实际float32 trace：events.tile_parameters的N=6、h=.20000000298023224、d=.2199999988079071；生产N*d=1.3199999332427979，再+.8为2.119999885559082，恰等float32字面2.12；相邻更小值为2.1199996471405029。因此原严格比较与实际TerrainSpec计算路径相容。
3. 初次失败夹具走“float32字面1.32再加.8”得到2.1200001239776611，与独立字面2.12不同。改float64只修数学单测夹具，不把这个独立舍入路径搬进产品。

当前events的平台比较实现与旧C1规范化AST相同，实际env.evaluate读取n*d而非独立字面1.32；已核对调用链。float32 trace是实际参数算术记录，float64单元是数学边界测试，CPU环境一步是构造/有限值验证；三者均不能冒称已在GPU策略下完成D22登顶或最终行为验收。

## 局部验证记录：实际19项对应GREEN

|记录|原stdout/exit|适用范围|
|---|---|---|
|A_GREEN_02.txt|11 passed、0skip/error、EXIT0，4.01s|spec+terrain|
|A_ENV_GREEN_02.txt|2 passed、0skip/error、EXIT0，6.25s；显式CUDA_VISIBLE_DEVICES空|真实CPU4环境构造/reset/一步、23tile及观察/指令/出生/schedule上界|
|cpu/b_geometry_routing_green.log|原stdout6 passed，0.78s；工具exit0依据B回执（该日志本身无command/exit尾注）|案例表/路由6项；B执行，reviewer未重复跑|

11+2+6是分次局部19项，不称一次完整CPU。源/测试全闭包尚未冻结，尤其B依赖还在变化，因此最终完整套件必须另在根统一冻结后执行。A角色采样8项归B，未将其计入A19。

已读取/保留报告中的10failed+2errors初RED、几何数学夹具失败、CPU warp默认device异常和routing真实FINAL21误上界的修复过程；没有skip吞错误。CPU第一次误用上游默认CUDA的设备异常不算GPU工程；显式隐藏CUDA后通过，未为此改contacts/物理。一个上游torch.jit.script弃用warning如实保留。

## 不能判定与后续检查

- B实际role_tiles是否严格用课程位置、完整validation/regression_policy队列/保护期、schema9顶层状态、stop4817全状态迁移、window64原件持久化及晋退恢复原子性，不在A1包中，均待独立B审阅。
- spec保留旧SUBSTAGE_REGRESS_STREAK=2及MIN_LEVEL .8/4/200兼容常量，同时native合同明确D/L训练EMA只诊断并给新regression_policy3次/1000保护。不能因常量仍在就声称D/L运行旧路径，也不能因A1通过就声称旧调用已删除；实际调用链必须由B证据闭合。
- A2 worker/validate在写，当前source_complete仍可能false；实际所有script/公共依赖闭包、最终contract/test清单须根统一冻结后核验。
- 没有完整相关CPU/GPU、512+8实训、23tile资源测量、真实loaded/ready首/二load、45baseline、30保持或最终物理300例证据；无新训练效果判断，无成功率额外准入条件。
- 整体工程最终放行仍须固定同源CPU/GPU/资源/成本与独立MD/JSON accepted=true/blocking0，主协调负责，不以本模块review替代。

## 绑定SHA

A owner清单：28199ba38d49e6fd9317b1bb15f31e730df86f2d5a784471f57ba75a0fea889e。
A实施报告：4c230f9336a22dd997abda625d39a3f3eb177eecd26fe3e181b02b7851487772。

|A1文件|SHA256|
|---|---|
|src/tasks/velocity/v07/spec.py|4a19735397b11bd3f9c25cceed6c47f3b7babcdc45a9cb64e7465ca991de116b|
|src/tasks/velocity/v07/terrain.py|db043e09d3d47844935a63df0934bad34992ae5b41a6d4981ea058c2da881c58|
|src/tasks/velocity/v07/env_cfg.py|0f8be2bbb652ba0dd7572008a5b43e7589a48a5c88861c2b8744e72da1927cbc|
|src/tasks/velocity/v07/env.py|e40cf56cbeb276e6f5cc3054f5bf3ebfea1c6a48edb0e0bb89eb73d4882f01f2|
|src/tasks/velocity/v07/map.py|14719bb47efe6519ca619b27af314355cb6b227c087e13b5c548138ab554658a|
|src/tasks/velocity/v07/contacts.py|ff0e5c1b8b991633f16d151a2f27bc8bcea66092f6278c5650e627d5749234e0|
|src/tasks/velocity/v07/events.py|076ed02c09bbd443c456e5df40f00f2684de88b9454fda91a1d7fa8fa02936c0|
|src/tasks/velocity/v07/speed.py|d6f8caab4b77503e68f675a78b5492d606155e1363cdc0f67499c290128b54e7|

|测试/证据|SHA256|
|---|---|
|tests/test_v07_spec.py|95c839914b94a3c5bdca31052c784fcdc85aadd7ddc724c354b23d8ca22795e7|
|tests/test_v07_terrain.py|bd4a0875e371bc2708b17db65ae0ad206253053b97a31a86d06ea0afc966cbcd|
|tests/test_v07_environment.py|dccc2a4b39684296b3b9d4ed7ddaeff9259b69434dc79445806317ffed50228f|
|tests/test_v07_geometry_routing.py|419137c5e4632fcc9b2604e2e6bb7f3af2566d554f82e9683087c19c5c44b8e0|
|logs/a2_ideal_stairs_v07/implementation/coordination/A_GREEN_02.txt|270cd04737457e5623aa60cb9bb15264246b89e3e41a312d72cec3b4f3f078d0|
|logs/a2_ideal_stairs_v07/implementation/coordination/A_ENV_GREEN_02.txt|7fa0656bddc77dc9584d44e4104126ca12f870745d84619cb45e830f0d1adf32|
|logs/a2_ideal_stairs_v07/implementation/cpu/b_geometry_routing_green.log|c706c44da365ed4dd50a8b473b274867d633c3c0bebb74c9ac39bb8ef80d7afa|
|logs/a2_ideal_stairs_v07/implementation/coordination/A_FLOAT32_BOUNDARY.txt|51f91850b82512644d9806721f82a22f41991b85e84151d4e60f4c89bf517e7c|

本批修改：只新增本独立审查文件；目的为限定A1 spec/quality双审和证据边界；验证为完整diff阅读、8+4文件SHA、旧保全SHA、静态AST与原始日志读取。未改产品/tests、未执行重复套件/GPU。共享logs/codex_changes.md由根按本批短报统一追加。
