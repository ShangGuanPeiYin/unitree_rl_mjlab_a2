# A2 v0.6 C1 续训 Implementation Plan

> For agentic workers: 使用 superpowers:subagent-driven-development 或 superpowers:executing-plans 逐任务实施；主对话协调，执行子agent实施，独立子agent验收。模型统一 gpt-6.1-sol，reasoning effort=high。用户已批准本方案并明确要求开始执行，不再重复询问实施或开训许可。

Goal：保留update4224已学能力，只收窄速度到0.3–0.6m/s、扩大D阶段预算，完整恢复模型及优化器继续学习。
Architecture：延续v06模型/schema8与原训练流程，通过绑定唯一停止checkpoint的C1迁移入口生成新合同和独立续训段。旧冻结文件/证据归档，不伪装为原合同恢复。运行参数改变限本文授权项及其必要恢复接线。
Tech Stack：现有Python/PyTorch/mjlab/MuJoCo、/home/xxs/anaconda3/envs/a2/bin/python。
Spec：原doc/sds/SDS_A2_IdealHeightMap_Stairs40_v0.6.md及logs/a2_ideal_stairs_v06/implementation/coordination/Q001–Q008裁决；本文在速度、预算和指定停止源重开方面优先。
批准依据：用户明确说“要求训练的爬楼梯速度为0.3-0.6即可……不需要改动奖励函数等配置，可以扩大预算继续训练”，并批准让A2upstairs执行前一轮具体方案；再明确工作流程与此前相同，执行方由DSH换成此Codex对话。

## Global Constraints
- 项目/home/xxs/dog/unitree_rl_mjlab_a2；先读取根AGENTS.md（若存在）及/home/xxs/AGENTS.md。所有修改追加logs/codex_changes.md，不覆盖历史，不commit/push。
- 新执行主对话：设置对话名为A2upstairs，thread=01a1e7eb-6ed5-79e2-b8ec-d70c50e4f2f5；主决策Codex=01a1d49c-5f5d-7890-ac95-9234b307139a。停止向DSH派工，不恢复R2seed43。
- 模型、奖励公式/权重/事件支付、PPO损失/学习率调整规则/探索/网络/观测/物理/安全终止/地形/晋退规则均不改变。严禁顺便增加D20保护期、延缓退级、改变EMA/64回合窗口或跳过D25。
- 512训练+8评估；rollout48、epochs5、minibatches4；每完整更新24576控制样本、98304物理子步、20Adam；控制dt0.02、物理dt0.005。
- 原例行save100、gate100、diagnostic50、retention200和评估前pending checkpoint机制保留。原资源门槛7168MiB整卡/5120MiB进程/4096MiB主机可用/10GiB磁盘保留；单GPU worker串行。
- Q008显示语义缺陷用sidecar解释；RNG完整隔离not_established限制继承。不得借本续训实施生产RNG修复或奖励改造。已停止run与旧control原字节保留。
- 关键不明点向主决策对话提问；独立工作可继续，不擅自推测。验收通过自动训练，不因已批准事项再次询问。

## Review Focus
1. 停止源必须限定SHA与budget原因，不能开放所有gate_failed/user_stop恢复。测试见任务2。
2. 当前Adam学习率实际为1e-5，不能被CLI初始3e-4重置；optimizer/normalizer/std必须保持。测试见任务2/4。
3. 新速度影响训练、门禁、诊断、两保持集、最终测试及所有硬编码断言；漏一个仍可能0.8或0.2采样。测试见任务1。
4. generation更新只清当前课程窗口/资格，不抹历史计费和已完成阶段；首次4300不足100新更新不能晋级。测试见任务2。
5. baseline/smoke/probe绝不作为续训权重或正式门禁；resume_loaded/ready必须实际可重载。测试见任务3/4。

## 1. 精确数值合同

### 速度（唯一行为目标调整）
- 所有后续学习阶段每episode固定指令vx~Uniform[0.30,0.60]m/s，vy=0、wz=0；原独立随机生成机制不变。
- COMMAND_INITIAL_VX=(.30,.60)、COMMAND_FINAL_VX=(.30,.60)；V_RAMP_UPDATES仍1000，但插值两端相同，所以整个V范围不变。V阶段仍至少1200、gate从阶段更新1000后计，R仍至少600、gate从400后计；不删除V或缩短阶段。
- 完整门禁、诊断、flat/H00保持、最终测试速度统一(.30,.50,.60)。保持每速5共15；普通每速20且>=18；最终每速100且>=95。所有阈值与晋级连续次数不变。
- 固定case seed映射与生成顺序不变；新速度生成新表SHA，旧速度表永远是旧证据，不混用。出生范围、R扩张范围与ramp不变。
- Q003平地距离公式D_net>=5c-1e-9，三档阈值1.5/2.5/3.0m；一致性atol1e-6m/rtol0保持。实际速度仍报告，不增加实际速度硬上限或超速惩罚。

### 预算（已批准扩大）
| 项目 | 原上限 | C1上限 | 已消费 | 新剩余 |
|---|---:|---:|---:|---:|
| D25 |1500|4500|1500|3000|
| D20 |1500|4500|24|4476|
| D合计 |3000|9000|1524|7476|
| 活动模型分支合计 |30000|36000|4224|31776|

其他宏阶段仍F0=1000/H=12000/L=6000/V=4000/R=4000；L12=1500/L20=1500/L40=3000。phase之和36000。原各子级累计计费不归零、不互借；达到原晋退标准即推进，不要求用满。
正式物理计算累计上限36100=最终活动分支最多36000+trial01损失50+Q005丢失分支50。原活动4224已包含其中，不重复加。当前实际累计4324，新增学习从4225起计；工程验证/零学习评估单列。
总预算剩余31776是全局上限，不承诺已结束F0/H的未用预算能借到D/L。

## 2. 指定恢复源和逐字段迁移
源：logs/a2_ideal_stairs_v06/train_seed42_trial02_resume_000400_q007/checkpoints/stop_004224.pt。
SHA256：108e03dc64effa94d2824bbf36842aee8f9c5fd9c2122be391b337706f4a51b9。
主Codex已CPU读取：learning.update4224、optimizer_step=adam_steps84480、control_samples103809024、physics_substeps415236096、learning_rate=1e-5；phaseD/substageD25/level_tick100/frontier_tile17/generation67；spent F0=200/H=2500/D=1524/L=V=R=0；D25=1500/D20=24。该事实也纠正Q009报告中“Adam4224”的错写，不改原报告，只追加勘误。
源停止原因gate_failed/substage_budget_exhausted，curriculum.finished=true/stop_status=budget_exhausted；pending_boundary/pending_update/incomplete_update均None；不得从更早好模型挑选恢复。

新合同标识：continuation_id=v06-c1-speed030060-budget36000；recipe=0.6-go2refs-c1；schema仍8，architecture保持v06-route-state-v1，12动作与模型键形状不变。code_version可标0.6.1（只表示续训合同修订，不暗示奖励升级）。旧身份保存在parent/source_provenance，不伪造旧模型由新合同训练。
唯一生产目录：logs/a2_ideal_stairs_v06_c1/continue_seed42_from004224_trial01。目录已存在拒绝fresh创建，不删除复用。

迁移行为必须在独立的新state副本上完成，输出migration_diff.json并绑定旧源/新冻结/本计划SHA：
- 保留所有model tensor、std、Adam状态和参数关联、normalizer、学习/物理/Adam计数、实际lr1e-5、训练及sampler RNG。ETH迁移元数据深拷贝并保留原来源，不能重新从ETH初始化或零化已经学到的route融合层。
- 只对上述精确源和停止原因允许在新state清除停止标记：curriculum.finished=false、stop_status=None；ledger.stop_state/stop_reason/error_type清空、lifecycle=running；新段stop_evidence不携带有效停止指令。原stop.json/control/checkpoint保留不变；历史停止事件原样作为历史，禁止重放其终止动作。
- phaseD/substageD25/level100/frontier17、spent全部保留；上限替换为第1节新合同，不能把已消费计数清零。
- 为速度分布变更调用与原_new_generation等价的重置一次：generation67→68、generation_updates0、ema={s:null,f:null,q:null,batches:0}、FIFO=[]、ready/hard/gate/minimum_level_failure streak=0、cooldown=5、window_metrics={}。清runner当前晋级streak/pending_transition，旧gate_log/gates记录保留但不能再计晋级资格。模型/奖励/优化器不重置。
- flat/H00保持资格保留：qualified_at200/400、last_completed4200、next_due4400、当前失败streak各0；新增速度表身份，历史不改。两个集合在4400仍按原计划检查，不从续训时刻重排计时。
- F0/H已完成的资格保留，不重训；当前D25需新generation至少100更新和连续两次合格门禁，不能复用旧D25通过。4300仅新增76更新，不能晋级；若无其他generation变更，最早4400/4500取得两次有效资格。
- 源无pending边界，不重跑4224；新episode恢复，非物理逐位连续。首个新PPO为4225。rollback_from=resume_from=4224、lost_updates=0、replay_billed_updates=false。
- 在任何学习前保存并正式重载验证resume_loaded_004224与resume_ready_004224；比较上述不该变化的tensor、optimizer、lr、计数，明确哪些课程/合同字段按批准改变。二次恢复新checkpoint不能再次generation+1或再次解除停止。

## 3. 文件与工作任务

### 任务0：冻结和分工
Files：新建logs/a2_ideal_stairs_v06_c1/implementation/下的冻结manifest、CPU/GPU/acceptance/coordination目录；保留原v06的所有冻结及停止证据。
- [ ] 先读取本文件、原SDS、Q001–Q008及Q009停机报告、录像README；重算源SHA，确认GPU无训练worker、停止任务不被其他执行方恢复。
- [ ] 记录原src/scripts冻结文件、当前git差异，归档准备修改的源文件；不得reset/clean其他用户工作。
- [ ] 主对话协调；执行owner负责参数/恢复接线；独立reviewer不能验收自己写的代码。主对话与子agent模型gpt-6.1-sol high，回执实际配置；不能只在文字自称high而无法核实配置。

### 任务1：速度和预算端到端接线
Files：src/tasks/velocity/v06/spec.py、speed.py、evaluation.py、orchestration.py及scripts/run_a2_ideal_stairs_v06.py/train_a2_ideal_stairs_v06.py中相关CLI/默认/断言；若值由导入单源提供，不做多余修改。
Tests：新增tests/test_v06_c1_contract.py，保留旧合同测试原语义，C1应显式版本化，不简单删失败断言。
- [ ] 先写测试：全部训练command bounds=(.3,.6)、全部评估三速(.3,.5,.6)、D子预算4500/4500、总36000、最终300例门槛不变；确认旧值导致测试失败。
- [ ] 采用一个明确C1配置身份使上述单源值贯通，不能只改CLI而实际env/spec仍旧。可增加隔离的c1配置/入口辅助文件，但不能重构整个v06；如必须改变model/reward/PPO算法正文先提问。
- [ ] 编码仅落实第1节；同步资源/账本/恢复/预检预算断言，不放宽一般校验。reward函数与权重保持原样；model与ppo仅允许身份常量必要接线，不改计算。
- [ ] 新case表按旧seed生成新速度，检验F0/H/D/L/V/R×validation/diagnostic/两保持/test适用组合，flat距离新锚点与H00固定目标正确；Q008保持记录语义仍按sidecar解释，不在此生产落地另一个候选修复。

### 任务2：一次性停止源迁移与完整恢复
Files：新增src/tasks/velocity/v06/continuation_c1.py；recovery.py/runner.py/orchestration.py与worker中仅必要入口接线。
Interface：continuation_c1.migrate_stop_4224(state, *, source_sha256)返回新的state及字段diff；验证唯一源、身份、计数和停止原因，纯迁移不读写源。继续按既有完整checkpoint正式加载，不只load模型。
Tests：新增tests/test_v06_c1_continuation.py。
- [ ] 先测试合法源精确变化符合第2节；任何错误SHA、user_requested_stop、behavior_regression、partial PPO或计数不符拒绝。一般resume仍拒绝任意停止源。
- [ ] 测试所有model/Adam/normalizer与当前lr1e-5保持，4224与84480计数不变；generation仅增一次；4400保持时钟不变；4300资格不足不能晋级。
- [ ] 实现受限迁移、来源链、两个4224文件保存/reload验证；缺ETH metadata在学习前拒绝，不再出现到150才保存失败。
- [ ] 测试真正裸模型→worker→runner加载→save→正式recovery reload；首次新更新4225，计数增为4225/84500Adam/103833600控制样本，不重计4224。

### 任务3：零学习基线
Files：新建logs/a2_ideal_stairs_v06_c1/baseline/及可复现入口/命令/manifest。
- [ ] 固定原4224模型，对D25/D20分别在0.3/0.5/0.6各5例，共30例；seed42，使用新validation表的每速前5例出生，标准范围x[-1.1,-.9]/y±.05/yaw±5。保存完整选定case表和逐例结果/失败原因，不使用只正中出生的录像代替。
- [ ] 独立评估模型/环境，确定性均值动作，无PPO；所有结果保留不挑选，不进入晋级streak/保持streak/课程EMA。基线通过率无额外开训门槛；仅工程错误阻塞。
- [ ] baseline不得污染待恢复的训练RNG或模型；新生产worker仍从指定源独立恢复。Q008隔离限制仍如实报告。

### 任务4：工程验收与开训
Files：implementation下CPU报告、smoke/probe、INDEPENDENT_ACCEPTANCE.md、C1_BRIDGE_MANIFEST.json、执行命令脚本；新增必要集成测试。
- [ ] 使用a2解释器运行与变更相关的测试及完整v06 CPU套件，期望0 failed；独立reviewer解释失败来源，不能以删除测试过关。
- [ ] 同冻结源码执行一轮原smoke（8env/H00/2更新）与原56更新probe（512+8env，5+50+1），保留真实50诊断、55→56裸模型恢复/loaded-ready/save-reload验证；所有工程更新独立计费，模型不进入正式续训。
- [ ] 独立reviewer核对diff仅速度/预算/身份及必要迁移接线；原reward权重/公式、PPO参数/逻辑、晋退条件、安全终止、随机化不变；源SHA与新闭包、baseline、CPU/GPU证据一致，blocking=0。
- [ ] 生成明确参数执行脚本：seed42、cuda:0、512/8环境、rollout48、epochs5/minibatches4、初始/最终vx范围均.3/.6、vy0/wz0、max-updates36000、phase budgets F0=1000,H=12000,D=9000,L=6000,V=4000,R=4000、save100/diag50/gate100、源checkpoint及SHA、受限C1manifest、新唯一输出目录。learning-rate配置仍原adaptive规则；真实恢复lr必须1e-5，不能被命令默认覆盖。
- [ ] preflight不创建生产目录；实际execute完成两个4224 checkpoint验证后自动从4225开始。用户已授权开训，不再请求重复确认。生产问题停机留证据并询问主Codex，不自动换seed/重开/加预算。

## 4. 开训后的汇报
首次回执：接单/实际模型与effort/分工/文档SHA。中期回送baseline三速×两地形结果、冻结与独立验收SHA。启动后回送当前lr、模型/Adam保持证据、loaded/ready4224 SHA、首个4225提交、实际样本与Adam计数、新run路径。
每次进度注明模型update、phase/substage、楼梯级数/阶高/踏深、各速度通过数、flat/H00保持结果、预算已用/剩余和累计物理成本。特别统计每次进入D20停留多少更新，是否仍4–5次就退回；只记录，不因此自行延长保护期或改退级阈值。
最终仍以40级20cm高/20cm深楼梯在0.3/0.5/0.6各100例>=95为行为目标，报告工程已知限制，不能以单次录像替代验收。

## 5. 权限和决策关系
A2upstairs负责执行与协调、子agent实施、独立子agent验收；任何需要调整本文之外训练数值、绕过来源校验或再扩大预算的问题，发到主Codex线程01a1d49c-5f5d-7890-ac95-9234b307139a。无问题直接完成整个实施→验收→启动流程。模型/权限无法按要求设置时先明确报告，不假称完成。
DSH不再是执行方；不要给它新指令。此文件和用户批准覆盖旧停止源“不得自动resume”的限制，但仅限本文件的精确来源及新预算合同，不改变其他停止任务的授权。

## 自检与当前验证状态
已覆盖速度/预算、恢复、基线、CPU/GPU验收、独立审核、自动开训和汇报。主Codex只CPU读取源模型状态并核实SHA与计数，未实施生产代码、未跑C1训练；效果尚未验证。原录像在logs/a2_ideal_stairs_v06/playback/update4224_D25_D20_20261117/。
