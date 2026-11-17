# 执行B进度与实现报告

## 绑定与边界

- 日期：2026-11-17。角色：B（主协调指定既有gpt-6.1-sol/high执行owner，不另作runtime模型断言）。
- SDS完整读取并实核SHA：303ae40a5c5e3fe82f9f0df3aae8d153136e016439087c8db9701aeb17197aca。
- 唯一源stop4817实核SHA：a41963a8bf5e6f1d4810b0a0c912ac65747d61bd481ea3432ef638a99e076e64。
- 不改旧v06/C1源、测试、旧SDS或checkpoint；不启动GPU/正式训练/push/子agent。共享codex_changes由根统一写；本报告逐批记修改并短报根。

## 批次B01：namespace handoff

- 目的：建立v07独立导入边界，禁止依赖C1环境变量先激活再patch。
- 文件：21个v07普通模块机械副本、scripts/a2_v07_bootstrap.py、tests/test_v07_namespace.py。
- 实际修改：机械转换模块/package路径v06→v07及Python公开类V06→V07，保留网络ARCHITECTURE=v06-route-state-v1；未复制c1_contract/c1_runtime/continuation_c1三个旧激活/迁移模块；复制spec移除末尾C1 opt-in逻辑。原模块计算路径暂机械继承，尚不代表native合同和v07行为实现完成。
- 验证：CUDA_VISIBLE_DEVICES='' OMP/MKL=2，a2解释器pytest test_v07_namespace.py -q -p no:cacheprovider。先2failed（缺bootstrap），后2passed/0failed、0.06s、exit0。raw：cpu/b_namespace_red.log、b_namespace_green.log。测试实际清洁子进程导入，非法A2_V06_CONTINUATION不影响v07，无v06/torch隐式导入；正常导入的旧src.tasks拒绝被替换。
- A文件已正式移交：spec/terrain/env_cfg/env/map/contacts/events/speed；B此后不编辑上述8文件。A负责将临时机械spec一次原生化为0.7/schema9/23tile/48000合同，B不末尾覆盖常量。
- A/B接口：CURRICULUM_TILE_ORDER不含flat，H00..H15,D25,D22,D20,L12,L20,L40；PHYSICAL_TILE_COUNT23，FINAL_TILE_ID21。B role_tiles按该序列取near/easy相对位置。

## B生产接口预告（待实现）

- regression_policy.py：只消费当前已持久完整validation payload（60首records、三SHA、模型/source/合同/task身份），纯决定protected/continue/regress/stop_lowest，state顶层schema9保存rolling3队列与最后事务键。
- continuation_v07.py：唯一4817纯迁移，checkpoint自身RNG/源state保留，gen72→73一次，D22预算键0，源终态/legacy marker留parent evidence；native普通load不重迁移。
- runtime.py及scripts：run/train/evaluate/validate/test_a2_ideal_stairs_v07；validate直接实现五GPU工程stage但B不执行GPU。
- 复杂事务：评估单结果缓存与sidecar继承Q010；课程决策与window64消费将在完整checkpoint中有幂等锚，不能在保存失败恢复后多消费窗口/多退级或跨phase重用旧评估。具体接口和故障验证后续追加。

## 当前状态

B01轻量CPU已GREEN；native spec由A实现中。B余下policy/课程/迁移/恢复/runner/validation尚未完成或未测试；不声称工程验收或行为改善。最终完整CPU/GPU由根统一freeze后触发。

## 批次B02：课程排序采样与D22评估合法性

- 文件：v07/curriculum.py role_tiles、v07/evaluation.py target_tile、tests/test_v07_curriculum.py（采样8项）。
- 差异：near/easy用A的CURRICULUM_TILE_ORDER相对位置，D22 near16/17、D20 near17/22、L12 near22/18、L40 easy含22截至18；保留物理FINAL21与原分配比例。未知物理frontier拒绝。evaluation目标合法性改查TerrainSpec表，不再用FINAL21当最大物理ID。
- 验证：a2/CUDA空/OMP-MKL2，pytest test_v07_curriculum.py；RED5failed3passed（旧ID算法），首次补丁文本定位未应用，b_role_tiles_green.log仍5failed并保留；修补丁定位后b_role_tiles_green2.log 8passed0.05s/exit0。A提供A_ROUTING_01.txt的真实D22失败后，B修target_tile，tests/test_v07_geometry_routing.py当前6passed0.78s/exit0（b_geometry_routing_green.log）；A两项早期夹具误断言由A独立解释，不删除断言掩盖。
- A真实CPU scene2passed/0skip已结束释放，B没有跑GPU。

## 批次B03：严格完整validation谨慎退级纯policy

- 文件：v07/regression_policy.py、tests/test_v07_regression_policy.py、tests/v07_test_support.py。
- API：new_state(generation)、reset_generation(state,generation)、validate_state(state,generation=None)、apply_validation(state,payload,expected_identity=...,generation_updates=...)返回(new_state,decision)。数字统一取A的REGRESSION_POLICY_CONFIG，不在模块另定阈值。
- 输入：真实CaseTable+60首terminal records，evaluation_records重算case/records/summary三SHA，额外核首terminal/env槽、tile、steps、outcome和逐槽物理成本；expected身份含当前model/optimizer/source/contract及任务。缺失/损坏/旧generation/diagnostic/retention拒绝，不当零成功。
- 决策：gu<1000 protected不积队列；S>12清；低成功rolling最近3，同代/任务且相邻100；Δ>=6继续但保留rolling，Δ<6按D20→D22→D25或L40→L20→L12退/最低停。最近事务完整fingerprint去重，冲突同key拒绝。队列/最后应用receipt有结构校验，无磁盘反查或模型运行。
- 验证命令：CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/xxs/anaconda3/envs/a2/bin/python -m pytest tests/test_v07_regression_policy.py -q -p no:cacheprovider。
- 原stdout：cpu/b_policy_red.log（24failed，module缺失）；cpu/b_policy_green_attempt1.log（24passed/0failed、11.60s、exit0，既有torch.jit弃用warning）。覆盖字面序列/保护期/rolling/间隔/完整坏证据/重放冲突，无模型构造或真实scene。
- 当前仅policy单元GREEN，manager/schema9/runner和故障事务尚未接好，不构成整体验收。已提交reviewer早审，暂稳定policy接口。根独占recipe parity测试，不重复造同类测试。

## 新owner交接与恢复接口

- A2正式接管train_a2_ideal_stairs_v07.py、validate_a2_ideal_stairs_v07.py及worker/工程协议测试，B未创建/修改它们，之后不编辑。
- B runtime.py：prepare_runner(contract,output_dir)、build_worker/main；CLI --v07-manifest固定implementation/V07_BRIDGE_MANIFEST.json，acceptance/INDEPENDENT_ACCEPTANCE.md；唯一source checkpoint/sha断言。
- continuation_v07.py：SOURCE_PATH/SOURCE_SHA256、load_designated_source、tree_digest、migrate_stop_4817。
- Qv07-001已明确：restore checkpoint RNG→原schedule一次→原PPO reset一次，接受变化槽额外随机消耗，不seed/倒回/拼result；将用runner.v07_initialization_audit与recovery/v07_initialization_004817.json。A已把ruling SHA1f84b142b70cc952eeca6e0a630f76d5d28f3d3b542cf2bcf5084d19be94a3cf纳入native frozen documents。

## 批次B04：课程与64窗消费接口

- 文件：v07/curriculum.py、tests/test_v07_curriculum.py。D/L保留64窗EMA诊断，移除EMA触发退级/最低级停；H原逻辑保留。完整validation传入policy，promotion优先于谨慎退级，预算最后检查；仅bool门禁不得代替完整60例证据。generation变更清当前队列而保留最后receipt。
- 64窗先deepcopy原始64行并调用window_sink，成功才pop FIFO和消费EMA；sink失败恢复manager逻辑前态。生产持久writer及runner原子提交仍待实施，此处接口不能单独作为故障恢复验收。
- CPU：b_curriculum_policy_red.log 5failed9passed；b_curriculum_policy_green_attempt1.log 14passed/8.05s。保护期前5更新不退、D25四坏窗只诊断、正常两门晋D22、bool-only拒绝、真实文件写失败不消费、预算仍停均覆盖。

## 批次B05：唯一4817迁移与原生schema9

- 文件：continuation_v07.py、recovery.py、model.py、runner.py checkpoint参数、tests/test_v07_migration.py。唯一源SHA和完整树摘要双绑定；仅深拷贝迁移，schema9/native合同、gen72→73一次、D22spent0、当前资格/FIFO/EMA清零，旧stop/legacy transition放parent。保留model/Adam所有group和参数ID、normalizer、learning、checkpoint自身六RNG及ETH原始证据。源lr7.593750000000002e-05未改。
- schema9显式顶层regression_policy_state；模型只接必要原ETH身份兼容，不改计算/初始化；source_block原SDS路径改spec.SDS_PATH，避免nativeSHA/旧path不一致。原生runner恢复和runtime仍待后续接线。
- CPU真实source：b_migration_schema_red.log先1failed7passed（schema尚缺字段）；b_migration_green_attempt1.log 8passed/17.66s，wrong SHA/userstop/partial/ID关联/RNG/metadata/counter反例拒绝，源不变且重复迁移拒绝。

## 批次B03-R1：独立审查恢复state反例

- reviewer真实反例发现：旧last_decision冗余缓存可篡改为regress，last_applied与queue仅比较update不足。修改前policy精确SHA a38362d5e28712bf439dfe5f936ffc7c88326b70b05d568021a67d7b54fafd20及tests/helper快照保存B03_R1_BEFORE；独立证据/tmp/a2_v07_b03_independent_counterexamples.json保留。
- 修复：删除持久last_decision动作缓存，state仅保存最后事务键与全payload fingerprint，duplicate固定continue/targetNone且不改state；完整六字段queue末identity与last_applied绑定，旧代receipt只允许队列为空且不冒充当前任务。未知冗余字段严格拒绝。真实应退级动作的未提交切口必须由runner bounded curriculum_transaction保存poststate/events续完，不能凭duplicate continue开启下一PPO；该集成仍待验证。
- 当前policy SHA c0ae8ca976501d4050f9377222b47f743c4d88c9d0c923dddee6cdaef208c8e4；限定diff B03_R1_FIX.diff。
- CPU：新增3项先b03_r1_red.log 3failed/27deselected，修后policy+manager+migration b03_r1_related_green.log 52passed/0failed/0skip、13.57s（1既有torch.jit warning）。只声明局部通过，未声明manager故障恢复/GPU完成。

## 批次B06/B07：64窗原子原行与native真实裸load

- 新模块window_evidence.py严格校验64原始规范行、先前EMA/window序号、排序、role/frontier/tile/generation与recordsSHA；文件fsync+不覆盖原子link+目录fsync，失败保.partial，重试同内容幂等，冲突/损坏拒绝。runner为manager装真实sink，消费前落frontier_windows/{update}_g{generation}_w{index}.json，不采样不计算新阈值。A owner已将模块纳入spec静态闭包。
- 新tests/test_v07_window_evidence.py先10RED（缺模块），首次9pass1fail因NaN夹具在重算SHA时先抛，修夹具后10GREEN0.93s；所有原日志保留：b_window_red/green/green2.log。
- runner原机械C1恢复改为continuation_v07及v07_initialization_audit、recovery/v07_initialization_{update}.json；四时点六RNG+课程sampler摘要只读。load恢复schema9顶层policy后按Qv07-001 schedule→原reset，不再迁移/清队列。
- tests/v07_runner_support.py与test_v07_recovery.py使用真实源迁移、真实裸model+45非空Adam；只环境为CPU recording fixture，不宣称真实物理。首load逐树model/optimizer/normalizer/learning保留，schedule(gen73,phase2117)先于reset；二load保留synthetic历史gu1083与低成功队列和保持时钟。先真实RED不存在continuation_c1，后与窗口合11GREEN20.34s（b_native_load_red/green.log）。

## 批次B08：bounded课程提交与真实异常恢复

- runner.py新增单个curriculum_transaction：prepared含已计费旧课程+policy；applied含旧测量/model/optimizer/source/contract锚、新poststate/policy/events/64window绑定及digest。当前完整validation payload与原sidecar绑定从同次Q010 cache传入pending，清eval cache后仍不从日志反查。
- window sink或manager异常回已计费旧state，不回滚PPO；已经新代的applied裸load校验完整绑定后续提交，严禁再on_update/evaluate/PPO；无可证明应用的decided/orphan事务拒绝。恢复schedule的sampler额外消耗只从逻辑状态对比中明确排除，不倒回/拼接RNG。
- 仅门禁/已有转移等提交边界作完整curriculum checkpoint；普通无task/无events update不增加全量保存频率。cache/pending未清严格拒学习；回退/晋级events去重后完整checkpoint成功才清。未知/partial事务可留证但resume拒绝。最低级三坏评估stop_state仍curriculum_stalled，reason精确persistent_low_validation_success，保三条证据。
- tests/test_v07_runner.py：最初3RED中2夹具误用CaseTable.tile_id；纠正后b_curriculum_transaction_red2.log 3项真实RED。第一GREEN批1failed3passed是历史transition数量误计（旧源7次回退历史正确保留），修断言仅计6100，并强化真实run自动exception保存→正式裸load，不手工代替自动异常路径。b_curriculum_transaction_green2.log 5passed0skip/67.14s（runner4+recovery1）。覆盖新gen74保存失败不重apply、已持久64后报错不双EMA/计费、损坏poststate拒绝、最低级新理由停机且普通resume拒绝。
- 局部测试使用synthetic历史更新数字建立5900/6000/6100边界，不声称这些更新真的执行；model/Adam为真实源非空状态，实际GPU PPO仍待根统一冻结后验收。

## 批次B09：原生runtime与独立入口（首轮）

- 新runtime.py、run/evaluate/test_a2_ideal_stairs_v07.py和test_v07_runtime.py；train/validate仍A2 owner。runtime有独立native合同与唯一4817初迁移，不依赖C1环境变量；同版本--resume-run仅合法非终态schema9/parent/source合同checkpoint，新segment不重迁移，工程source-reload显式engineering_only不供正式恢复。
- 固定manifest格式schema整数9、manifest_schema=a2-v07-bridge-manifest-v1；SDS/源SHA/静态闭包/合同/逐文件SHA/独立验收MD+JSON均准入检查。CLI关键数值仍断言SDS，首目录固定，native后续新目录在v07log root；不execute只preflight。prepare_runner实际裸load/save_verified loaded/ready，保原model/Adam/normalizer/lr/metadata及parentchain。
- CPU：runtime8RED缺模块；入口首次收集错误缺pytest import已保留(b_entry_red.log)，修测试后3真实RED+2pass(b_entry_red2.log)；b_runtime_green1.log 13passed0skip/10.01s（CLI8+namespace5）。本批只纯CLI/入口验证，真实runtime env组合待下一批；未启动GPU/训练。

## 批次B10：当代资格隔离、严格初始化锚与独立停止优先

- continuation_v07.py将旧C1 evaluation.cases/latest和gate.records完整留parent.historical_evaluation/historical_gate，native当前cases={}、latest=None、records=[]；原旧源不改，model/Adam/RNG保留。对应新增真实source反例先RED后migration9+runtime9共18GREEN24.57s，b_runtime_integration_red/green.log。
- runtime.py真实常量修为spec.TRAINING_NUM_ENVS；manifest schema严格整数9。prepare_runner真实组合（RecordingEnv、真实模型/45Adam）已验证loaded/ready各独立bare reload aligned、源lr保留、工程ready正式恢复拒绝；原生新segment分支另实际验证gu1083/已有低成功队列不重迁移。
- runner初始化前事务sampler必须与checkpoint完整前/后态精确吻合；仅Qv07-001 load后记录的实际schedule/reset sampler推进可作为当前事务的额外消耗，不能泛化忽略sampler差异。新增b_transaction_sampler_red.log真实1RED；修后在下述18通过内GREEN。
- 已有保持/诊断stop及user/resource/monitoring独立检查先于课程promotion/regression，预算仍最后。每正常边界独立检查一次，之后只作课程/预算判断，不改阈值/资源限制。新增b_independent_stop_red.log真实3RED（保持/诊断/user），修后GREEN；不在已停边界消费坏评估或换代。
- b_boundary_integration_green.log为18passed1failed106.92s，唯一失败是新manifest测试lambda未接root参数的夹具错误；修夹具后仅该项b_manifest_green2.log 1passed16.76s，其余18项不重复跑。未将第一次报全GREEN。
- 生产evaluate实际60回调→sidecar安装后故障→run自动exception checkpoint→正式裸load续完同测量/退级，及native runtime现有队列/gu新segment恢复，b_actual_handoff_integration.log实际2passed43.92s，无再评估/PPO；fixture只合成历史与case行为，不声称GPU行为效果。
- 新result显式附regression_policy_state供停止/队列审计，checkpoint顶层状态保持相同；普通终态resume守卫保持拒绝。
- 发现原C1/机械native外部control.json未绑定runner.stop_flag，已报根并等待Qv07-002主决策；未隐含实施新pause/stop协议，也不以stop_flag夹具通过宣称外部控制有效。

## 当前待完成

B核心候选已交根限定只读review；policy R1已独立PASS，但manager/runner/runtime仍需独立审查、最终统一freeze完整CPU、五阶段GPU实测。B不启动GPU或正式训练。新增native Q010首episode与records测试仅从旧测试机械替换namespace，原测试不改，用于锁定继承语义，当前运行中，结果待记录。Qv07-002外部控制未裁决前不改其协议。没有行为改善声明。

## 批次B11：Qv07-002实际外部控制协议与收尾

- 修改日期：2026-11-17。新operator_control.py；接线runtime.py/runner.py/orchestration.py；tests/test_v07_operator_control.py、test_v07_control_runtime.py。A已将裁决dad458...与实际helper required项纳入spec，最后A spec SHA c79a0afec8daa1f8ab67e3e7d9991997c49d5f4c77a8ed83ef6891b6089e07c0。
- 文件action仅continue/pause/stop，reason或resume_authorized不能代替action；missing默认continue。首次有效pause/stop即锁存，之后continue不能撤销；pause终态user_requested_stop/operator_pause、pause_requested=true、resume_authorized=false；stop同用户终态user_requested_stop。原生终态普通resume仍拒绝。
- publisher原子替换control，解析失败/非object/缺或未知action/读取异常具名ControlProtocolError。原始字节按SHA另存bin和base64审计，错误类型写自动exception checkpoint；读取失败如实无raw，不伪造原件。监督器不再用failed覆盖用户control，真实终态优先且不在其后重读控制重分类。
- runner在run处理pending前、每次新PPO前、完整PPO计费后处理评估前和完整边界后轮询；collect/Adam过程中不杀。命中后保存退出，不resource_guard、不final GPU评估；已有独立终态不重写。更新中请求让本次20 Adam完成并计费，待课程/评估pending保留，绝不伪curriculum_decided或下一PPO。native prepare中的pending完成也走同轮询/自动异常留证。
- 纯helper9RED→9GREEN0.45s：b_control_helper_red/green.log。监督器覆盖原件反例先真实1RED8.31s：b_control_supervisor_red.log。实际runtime七项先7RED88.54s：b_control_runtime_red.log；接线后helper10+真实runtime7共17passed/0skip81.70s，b_control_integration_green1.log（session61410实际exit0）。
- 最后补有效请求的证据IO失败锁存反例：b_control_latch_io_red.log真实1RED；改为先锁存raw与action，再持久证据，重试只用原raw，不接受后续continue。与全部11helper、确切window绑定、Q010真实CPUenv单项共13passed0skip19.70s，b_control_window_q010_green.log/session73072实际exit0，SHA54655eb8f6325c3a3b4f81d7cc02547e00fe45c8441f84c9049875e6d64972a6。17整链是早期helper版本，最后小改有11helper当前版本覆盖，统一冻结全native尚待根执行，不能混称完整同SHA验收。
- 64window最后交叉绑定：b_window_transaction_binding_red.log真实1RED21.78s，合法其他update窗口即使三处SHA重算也不得代替before_state FIFO前64、前EMA、该update/phase/sub/generation/window序号。已在最终13GREEN中覆盖。未改消费判据。
- native Q010两派生文件首次50passed1failed11.47s（b_q010_native_parity.log），唯一错误是复制测试仍import V06Env；仅修为V07Env，真实CPU2env逐步sum_vx/episode_length_buf、sum_speed/episode_length_buf及reset清零单项在13GREEN通过。旧Q010文件未改，其余50同内容不重复跑；全native最终套件由根重验。

## 完整B实现逻辑与关键参数

本批是原生0.7/schema9迁移和谨慎课程实现，尚非行为效果验收。唯一源stop4817按实际SHA与完整树摘要绑定，保全部已学model/std/Adam45参数ID关联及完整group、normalizer、ETH原metadata、checkpoint六RNG，继承真实lr7.593750000000002e-05；不从result拼RNG。首次仅gen72→73，gu0，首新更新4818。Qv07-001明确恢复RNG→原schedule一次→原PPO reset一次，允许变化槽额外初始化消耗，四时点只读摘要；不seed/回卷、不声称物理位精确。native后续新segment保已有gen/gu/退级队列/保持时钟，不再迁移。

训练仍512环境、评估8，rollout48、5 epochs×4 minibatches=20 Adam/完整更新；每更新24576 control samples、98304 physics substeps。保存/正式门禁100、诊断50、保持200。源已计4817不重学；全分支cap48000、正式累计含旧100损失48100。D25/D22/D20预算6000/6000/9000，源spent2083/0/34不清；宏D21000已spent2117。账面剩43183，但已完成F0/H未用10300不转移，之后最多新增32883，预算不是保证训练能跑满。

课程保F0→H→D→L→V→R；D插D22并按A的难度序列取near/easy，物理新tile22不替最终L40/tile21。H原EMA升降与最低规则不变。D/L原64窗FIFO与EMA保诊断、原行先落盘；原EMA退级/最低停调用取消。新代gu<1000只禁止性能退级/最低停，不屏蔽独立停止、预算或正常晋级。退级只吃同次完整validation60例（三速各20）首回合与table/records/summary三SHA、当前模型/optimizer/source/合同/代/任务；S<=12入rolling最近3，相邻100同代；S>12清，Δ>=6保rolling继续，Δ<6才退D20→D22→D25或L40→L20→L12，最低D25/L12停curriculum_stalled/reason persistent_low_validation_success。若源一直留D25，5800 gu983不入坏队列，最早5900/6000/6100；正常每速>=18/20且原门禁资格满足时仍可较早两门晋级。

命令沿批准合同恒vx[.3,.6]、vy/wz0；评估.3/.5/.6。原成功/失败阈值、F0第500窗、保持flat/H00各15例与独立两坏保持停、最终每速100共300且每速>=95均不改。Q010每case首terminal或F0未终止500窗口深冻，真实case步数与批量额外物理计算分账；不从后续自动reset回合补证据。

奖励项、权重、时间累计、网络/观测/物理/随机化/出生/探索投影不改。PPO目标保持原clip .2的min surrogate，加原clipped value loss系数1、减entropy .003；gamma .99/GAE .95、全模型grad clip1、std投影[.05,1]、KL阈值.005/.02和lr×/÷1.5及[1e-5,3e-4]夹紧不改。源当前lr不以CLI initial_lr覆盖。数值对拍由根独立recipe parity负责；本报告不把工程通过或新保护期解释成爬楼效果改善。

## 最终交接边界

全部Bowner源/测试及原始RED/失败修正/GREEN日志按TASK_B_HANDOFF.json逐文件SHA绑定。TASK_B_FULL_DIFF.patch以before/source冻结原件或/dev/null为基线，包含全部源和tests；TASK_B_FUNCTIONAL_DIFF.patch仅给namespace归一后的便读视图，原始beforeSHA和全diff为准，不代表删改旧文件。共享logs/codex_changes.md仍由根统一追加（最终对用户链接该文件）。

仍未验证：最终统一冻结同SHA全native CPU、五阶段GPU、真实512首二恢复缓存/角色/RNG观测、D22资源/恢复probe、45例baseline和30保持协议及行为效果。B不启动GPU/正式训练，不commit/push/派子agent。独立核心review和最终验收由根转审；之后产品暂稳，只接具体finding，不自行延伸功能。

### 最后既定duplicate拒绝防线

- curriculum.py仅增加两行：manager在无applied课程事务的决定调用中收到policy duplicate即抛PolicyEvidenceError，不把其continue当成已完成动作。合法applied恢复由runner保存poststate/events完成，不再进manager；纯policy R1字节和SHA保持不动。
- test_v07_curriculum.py新增已消费当前receipt但manager尚无应用证明反例；b_unproven_duplicate_red.log真实1failed8.34s，修后b_unproven_duplicate_green.log实际1passed/0skip8.26s（session66581，exit0）。异常恢复逻辑前态和policy不变；没有新阈值或额外功能。后续未覆盖的结构/事务组合留独立review与最终冻结suite，不扩大自选实验。

## B CORE R1/5：两项确认P2的一次限定修复

修改日期：2026-11-17。完整读取TASK_B_CORE_REVIEW.md（SHA9ca6bdf73b5019b11bbeada6bcd35b16831485a2dec7d6b3668f4fd687cd3398）与TASK_B_CORE_R1_BRIEF.md；原owner处理F-B-CORE-01/02，不改纯policy、配方或A/root文件。B_CORE_R1_BEFORE逐项保存原36源/tests和原handoff/report/full/functional diff/review/brief；其MANIFEST SHA c3ac06445213bc52b5656caec14d8a510debba65960930be8a1d6db6dcbef699。原v1完整CPU219及旧587为根已完成的历史证据，不冒称覆盖本R1闭包。

### F-B-CORE-01：强制消费证明与前后状态一致

- 修改前：只校验windows列表中存在的每个条目；windows=[]可绕过实际原件验证，甚至beforeFIFO也遗漏而postEMA已消费仍接受。
- 修改后仅runner.py：prepared必含before_state+before_policy的SHA锚；applied必含实际consumed_windows及独立停止标记。windows数量必须等于实际消费数，并与前FIFO长度/完成状态/独立停止相容。未换代时后FIFO必须精确为前FIFO移除已证明64行，后EMA必须由这些实际原行和原系数解释；换代只允许+1且EMA/FIFO按原规则清零，所消费旧代原件仍必须存在和绑定。独立停止必须确实已生效，前后课程/policy无变化且无消费/events，合法windows=[]保留。
- 原有每窗文件/recordsSHA/旧update、phase/substage/gen/window序号/前EMA/FIFO前64交叉校验仍在。缺证据只拒绝，不重写原件、不从日志补造、不调用PPO/evaluate、不回卷RNG、不新增每update全模型保存。EMA复算仅审计，不应用或改变课程EMA。
- RED含遗漏唯一原件及windows、前FIFO也遗漏、前SHA重新计算，分别覆盖同代与正常晋级新代；合法无消费、正常消费、generation reset、独立停止四项对照保留。已有合法其他窗口替换拒绝仍在受影响suite。

### F-B-CORE-02：未解决有效锁存早拒绝native准入

- 修改前：同一helper锁存正确，但IO异常checkpoint的control_error.latched_request未带进新helper，后来continue能得到ready/学习准入。
- 修改后仅recovery.py.resume_contract：发现当前checkpoint terminal.evidence.control_error中仍有有效pause/stop锁存，直接RecoveryBlocked要求裁决。该检查由正式preload(mode=resume)执行，早于runtime新model/env/GPU/pending/ready，不读取新control来撤销旧请求，不写新的控制文件或改原异常证据。
- schema保存和mode=train只读证据仍允许；原stop_state=None/error_type=ControlProtocolError/raw/base64/SHA/exception文件均保留，不伪造用户终态。没有有效锁存的坏JSON异常仍可在外部纠正后按原非终态native恢复；没有新增pause自动resume协议。
- 实际runtime CPU测试覆盖pause与stop、普通与pending两入口，先真实bare模型/完整非空Adam和RecordingEnv制造_snapshot IO异常及自动exception checkpoint，再用新segment continue+resume_authorized=true恢复，必须在环境构造前拒绝。断言原exception SHA和原control字节不变，无ready，无pending评估/资源/final/PPO工作。另有真实坏JSON无锁存纠正后ready通过的对照，以及resume_contract证据不变/可读的三项对照。没有GPU或正式PPO。

### 实际命令与结果

- RED command：`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/xxs/anaconda3/envs/a2/bin/python -m pytest tests/test_v07_runner.py tests/test_v07_control_runtime.py -k 'test_r1_' -q -p no:cacheprovider`。
- 原日志cpu/b_core_r1_red.log：9failed、4合法对照passed、17deselected，121.69s；工具session53854实际exit1。SHA d00b4bad11d829464eeb6303f885457887baaa1cbf143904a0bc9ebdd2293900。
- GREEN command：`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/xxs/anaconda3/envs/a2/bin/python -m pytest tests/test_v07_runner.py tests/test_v07_recovery.py tests/test_v07_control_runtime.py tests/test_v07_runtime.py tests/test_v07_operator_control.py -q -p no:cacheprovider`。
- 原日志cpu/b_core_r1_affected_green.log：57passed、0failed/skip，331.61s；工具session44467实际exit0，日志自身含command和SHELL_EXIT_CODE=0。SHA 2aab73ccf38c783a47a9c171695833ddf16e2728efe42339b0b2c66c8904678b。仅既有torch.jit弃用warning。
- 实际变化精确5文件：生产runner.py/recovery.py；测试test_v07_runner.py、test_v07_recovery.py、test_v07_control_runtime.py。regression_policy.py仍c0ae8ca976501d4050f9377222b47f743c4d88c9d0c923dddee6cdaef208c8e4；其余31owner文件字节不变。

交接以TASK_B_CORE_R1_HANDOFF.json和TASK_B_CORE_R1_FIX.diff为准，原TASK_B_HANDOFF及原before包保留历史，不只改旧status复用证据。全B当前baseline diff另存TASK_B_CORE_R1_FULL_DIFF.patch。R1产品现暂稳，等待只针对findings+新diff独立复审。完整v2 native冻结CPU、五GPU阶段与行为效果仍未验证，由根在限定复审后协调；B未启动GPU/正式训练/推送或新子agent。
