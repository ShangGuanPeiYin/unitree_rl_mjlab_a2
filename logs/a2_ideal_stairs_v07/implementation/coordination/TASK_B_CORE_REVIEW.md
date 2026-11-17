# B核心限定独立审查

> 最新限定结论（B CORE R1）：Spec compliance / Code quality 均 **PASS_WITHIN_B_CORE_R1**。F-B-CORE-01与F-B-CORE-02均ADDRESSED，当前B核心blocking=0；下文初审REQUEST_CHANGES保留为历史。完整同freeze CPU/GPU仍待最终验收，本报告不是全局accepted。


日期：2026-11-17T14:22:38.522501+00:00；reviewer=/root/c1_reviewer，显式派工gpt-6.1-sol/high。依据GLOBAL_CONSTRAINTS、TASK_B_BRIEF、完整TASK_B_REPORT、已完整读取的SDS303ae40a5c5e3fe82f9f0df3aae8d153136e016439087c8db9701aeb17197aca及Qv07-001/002。审查范围为B稳定交付36源/测试，未增加reviewer、未实现产品或执行者tests、未Git/GPU/正式训练。

## 双verdict与全部确认findings

- **Spec compliance：REQUEST_CHANGES。** 主要native迁移/课程/单结果与bounded事务/control接线符合设计方向，但当前恢复证据缺两个P2闭合项，不能放行。
- **Code quality：REQUEST_CHANGES。** 确认P0/P1=0、P2=2，**B核心blocking=2**。F-B-CORE-01为消费64行的缺失证明可被空列表绕过；F-B-CORE-02为已经观察锁存的pause在证据IO异常后跨native恢复被continue撤销。
- 这是完整限定核心review，不是最终GPU acceptance。先前B03-R1纯policy限定PASS保留且当前SHA未变；本轮不是无因重审它，也没有重跑已通过局部套件。正常producer未被证明已产错记录或发生违规训练；两个缺口均由明确的异常/坏缓存反例证实。

## 包与原件实核

完整读取真正baseline差异TASK_B_FULL_DIFF.patch **3519行/207561字节**，SHA `97f61d8d9d8d6936d11bf469a27e209fd3b1fcb9216ea8fc19dc8c32d948d1ba`，包括全部源及tests。TASK_B_FUNCTIONAL_DIFF.patch只作为namespace归一便读视图，不当真实基线。handoff SHA `03d373bbc8e77a201728f28ede45a64678d78c8d1259b16e0a5864fc8bb92b4e`；全部36个当前SHA、所有非/dev/null的before源SHA、48个原始CPU日志SHA均实际重算匹配，完整diff target集合与36文件集合完全一致。独立反例前后36文件无字节变化。

实际A spec SHA `c79a0afec8daa1f8ab67e3e7d9991997c49d5f4c77a8ed83ef6891b6089e07c0`，包含window_evidence/operator_control依赖和两项补充裁决；Qv07-001实际SHA `1f84b142b70cc952eeca6e0a630f76d5d28f3d3b542cf2bcf5084d19be94a3cf`；Qv07-002实际SHA `dad4580188773b0345f3f741b5f8a3d9f988652d7a257c8e2205dcd1d23adbd1`。

根同时冻结v1供CPU不改变本结论：当前219项native完整CPU通过是根回执，reviewer没有重跑；测试通过不能排除下述已实际确认反例。v1明确core_review_pending，不能据此跳过修复/新freeze/相应验证。

## F-B-CORE-01 — P2：applied缓存可漏掉实际消费窗口及其原行证明

**位置：runner.py:1878–1886，_validate_curriculum_transaction。** 当前只限制windows为list且长度<=1，随后for循环验证“存在的每个窗口”；没有确保本次确实消费了窗口时必须存在对应记录，也未校验消费前FIFO/EMA与实际post消费变化的一致性。

已有正确防线是：若一个window存在，实际文件、完整绑定SHA、旧update/phase/sub/generation/window_index、previous_ema和beforeFIFO前64都会核验；B新增“替换成其他合法窗口”的反例已覆盖此防线。但是把唯一window直接遗漏为空，会完全跳过这些检查。

### 实际两个variant与精确影响

1. 用真实CurriculumManager消费规范64行，生成正常before FIFO64、post FIFO0、EMA batches0→1及真实write_window原件。当前validator先正常接受完整applied事务；随后只将windows=[]、删除/tmp窗口文件并重算post_sha256，validator仍接受。**这个variant的before_state FIFO中仍保留64原行，因此不声称其全部raw已经消失**；确认问题是强制消费绑定/原件验证可被空列表跳过。
2. 同一缺口再把before_state.episode_fifo原行也漏存为空：before/post FIFO均0、beforeEMA0/postEMA1、windows=[]且原件缺失，重算post_sha仍实际接受。此时缓存只剩EMA等聚合值，不能证明哪64条实际outcome被消费。before_state本身没有被post_sha绑定，当前验证也没有发现它与postEMA消费次数不相容。

这不是没有更新hash的损坏。外层post摘要和当前模型/source/课程身份对齐仍成立，当前validator明确返回applied；缺的就是应当被拒绝的消费证据一致性。原始36文件与生产证据没有修改，删除的仅/tmp新建夹具原件。

该验证器正用于正式load恢复applied事务；恢复随后直接提交poststate而不再调用manager。若坏缓存缺失窗口却被认可，可越过SDS5.4要求的“实际64outcome或不可变原件索引及hash”证明。此处独立轻量反例直接调用真实validator，未执行正式裸model load；不将方法范围夸大为该variant已跑完整runtime链。

**限定修复要求：** 原owner按R1先RED再最小闭合消费数量/前后状态/实际原件关系。没有消费时合法windows=[]必须保留；需要覆盖正常消费、漏唯一window、漏before FIFO与原件、已有合法窗口替换、generation变化后EMA/FIFO合法重置，以及独立停止未消费的合法情形。不得为验证而重跑PPO/评估、回滚RNG、改变课程或额外保存每update全模型。具体proof表达归owner，reviewer不实现。

## F-B-CORE-02 — P2：有效pause的锁存未跨ControlProtocolError异常恢复保存

**位置：runtime.py:94–106创建新ControlProtocol并注入stop_flag；runner.load:909–940恢复error_type/pending但没有恢复或拒绝原已锁存请求；operator_control.py:56–83仅内存锁存。** Qv07-002要求首次有效pause/stop一经观察即锁存，后续continue不能撤销，pause没有自动恢复授权。

当前同一helper对象已正确做到：先锁存有效action和raw，再持久证据；即使_snapshot失败，之后重新poll也只使用原raw。这由owner最后新增的IO失败单元测试覆盖。但新segment会创建一个空helper；该锁存只作为ControlProtocolError.audit.latched_request进入自动exception checkpoint，load既不采用它，也不阻止从该证据恢复。

### 实际真实runtime链反例

- 从唯一源迁移出的合法native checkpoint开始；实际runtime.prepare_runner，真实裸V07模型、完整非空Adam及save_verified链，只有env是RecordingEnv，telemetry/case runner用CPU替身；没有物理仿真。
- 外部action=pause已有效观察并锁存，随后注入证据_snapshot IO异常。真实runtime捕获具名ControlProtocolError，自动保存exception_004817.pt，原始pause字节和audit.latched_request.action=pause仍在ledger.terminal.evidence.control_error内；stop_state是None，error_type是ControlProtocolError。
- 换新native segment，control写成continue，从上述实际exception正式preload/prepare。当前代码创建了**verified resume_ready**；新helper.latched=None，_poll_operator_control返回False，stop_state仍None，_check_resume_evidence(stage='learning')实际放行。
- 全链update仍4817，**没有执行任何PPO**；确认的是已观察pause被后来的continue撤销并恢复了学习准入，不声称已训练过4818。

**限定修复要求：** unresolved已观察有效pause/stop的原始锁存必须跨异常checkpoint恢复继续生效，或在缺乏可证明处理时明确拒绝学习恢复；不可只靠当前control重新读，也不可抹掉原工程错误/原字节、给pause新增自动resume协议，或把所有可纠正的坏JSON异常不加区分地新建终态。具体采用恢复原锁存还是fail closed由根/原owner在Qv07-002边界裁定。真实runtime测试应覆盖pause与stop、原控制字节/错误留存、新segment continue不能撤销、pending入口同样不重评/不资源/不finalGPU、学习准入不开放；无需新PPO。

## 独立反例回执与验证范围

- `/tmp/a2_v07_b_core_independent_counterexamples.json` SHA `a72695bdf599f336fa610b0ebe54c7205a3575d3c3d519706bdd12b8215192f0`：F01 variant1和F02完整native runtime链，工具实际exit0。exit0表示成功复现并记录缺陷，不表示产品通过。
- `/tmp/a2_v07_b_window_missing_source_recheck.json` SHA `675450a7c7929e825cf241463b6a126a639a3031853de50e101ce03da1146d53`：同F01 variant2，工具实际exit0，未覆盖第一份原件。
- 两批都CUDA_VISIBLE_DEVICES=''、OMP_NUM_THREADS=2、MKL_NUM_THREADS=2；未启动GPU、真实env仿真、PPO、pytest套件或新reviewer。F02确实构造/加载真实模型和Adam，F01不构造模型，仅当前validator和规范JSON。
- 本轮完整包读完后才集中给这两项全部findings；没有继续扩展同finding变体或提出奖励/引擎结论。

## 其他核心路径的限定审查结果

### 原生namespace和配方保持

v07 bootstrap在首task import前建立隔离namespace，机械版本导入不先激活C1，也不读A2_V06_CONTINUATION。run入口委托native runtime，正式CLI不接受旧桥/fresh来源。ARCHITECTURE保v06-route-state-v1；ppo.py只有引用文案差异，rewards.py只有文案namespace，model.py只版本类名和保原ETH metadata所必需兼容。网络/奖励权重和累计、PPO数学没有本包新增计算改动；根已有独立真实recipe parity，reviewer不重复它，也不声称物理效果改善。

关键设置保持512训练env/8评估、48 rollout、5epoch×4minibatch=20Adam/更新，每更新24576控制/98304物理；gamma.99/GAE.95/clip.2/entropy.003、原value/grad/KL/std投影规则不变。源实际lr7.593750000000002e-05不被CLI recipe initial3e-4覆盖。commands恒[.3,.6]且vy/wz0，评估.3/.5/.6；reward/PPO/出生/终止不自行调整。

### 一次4817迁移和后续native恢复

迁移前同时验指定文件SHA和完整源state tree digest，拒其他stop原因、partial/pending、计数/Adam ID关联/RNG/metadata变更；深拷贝源，不改原文件。只schema8→9/native identity/source/预算合同、generation72→73一次及gu0，清当前EMA/FIFO/晋级资格/旧hard最低失败/终态/legacy pending。原旧门禁与evaluation表留parent history，不混入native qualification；原transfer metadata、model/std/full Adam groups+ID+momentum、normalizer、learning及checkpoint六RNG逐树保留。源D25spent2083/D20spent34不清，新增D22spent0；保持资格4800/next5000及streak0继承。

runtime先裸model/load，再独立save_verified loaded/ready；engineering ready不能作正式来源。native后续新segment走普通schema9/parent/source合同，不再调用迁移、不清gu/队列/保持时钟；普通stop/complete/partial PPO不可resume的守卫保留。Qv07-001的checkpoint RNG→原schedule一次→原reset一次被实际接线，四时点摘要只读，允许变化槽额外初始化；sampler差异只接受当前load实际记录的额外推进，未回卷或拼result。实际512槽/RNG/缓存仍待GPU。

### D/L谨慎课程和预算

role_tiles用A的独立难度顺序，不以物理22代最终21。D/L旧EMA/hard_streak只诊断，原64FIFO消费/EMA数值保留，H旧决策保持；不再以D/L训练窗退级/最低停。完整正式60例（三速各20）的当前模型/Adam/source/contract/phase/substage/gen/task身份与三SHA消费后才产生策略决定，bool-only和缺/错/foreign evidence拒绝。

当前代gu<1000保护不积坏队列，正常有效门禁晋级不受阻；S>12清队列，S<=12入同代相邻100更新rolling3，delta>=6继续且保rolling，delta<6退一级或最低停persistent_low_validation_success。已消费receipt重复只返回no-action；manager对没有applied证明的duplicate明确拒绝，不把continue误作完成动作。B03-R1模块字节仍c0ae8c...，既有限定结论未扩大为事务通过。

budget仍判断在合法晋级/谨慎决定后；D25/D22/D20 cap6000/6000/9000，宏D21000/global48000/formal含旧100为48100。源更新4817不重学，首4818；账面剩43183、不可挪F0/H余额时之后最多32883。若始终D25，5800gu983不能入坏队列，最早5900/6000/6100三次低成功停止；这不是保证会跑到该点。

### Q010单结果与bounded事务

真实评价结果在当前Q010事务中pack/deepfreeze，先原子sidecar+file/hash/主binding，写失败exception缓存同一完整result，恢复不重评；校验table/records/summary三SHA并重算summary，applied不双计。pending将validation完整payload与绑定直接交给当前课程，未从日志倒查旧summary。

课程prepared保存已计费旧state/policy/model/Adam/source/task锚；manager/sink失败恢复逻辑前态，不回滚已完成PPO。applied持有poststate/policy/events/windows和摘要，恢复从该poststate提交而不再on_update/case_runner/PPO，events去重；决定marker缺应用证明和orphan拒绝。pending/tx未清时learning guard拒绝；已有eval或transition提交边界完整checkpoint成功后才清，普通无task无events不加每update全量保存。F01是此设计的缺失窗口证明校验不足，不等于整个事务设计被否定。

窗口正常producer确实先deepcopy64并写fsync+atomic hardlink/dir fsync且实读校验，成功才pop/consume；writer保partial、同内容幂等/冲突拒绝，schema严格校验角色/地形/代/排序/结果和前EMA。F01不会被这些正常写入测试覆盖，必须在恢复缺失证明分支补齐。

### 独立stop与Qv07-002已正确覆盖的路径

现有独立保持/诊断终态、用户/资源/监控先于课程决定，预算最后；新低成功终态保三条证据，普通终态恢复仍拒绝，既有stop不被后来control重写。外部合法action只有continue/pause/stop；缺文件默认continue，reason/resume_authorized不能替action；pause/operator_pause、stop/user_requested_stop都failure_class=user且resume_authorized=false。

真实runtime注入stop_flag（不是只有假布尔stop测试），run入口pending前/新PPO前/完整已计费PPO后评估前/完整边界后有poll；更新中request不杀collect/Adam，当前20步完整计费，pending未完成证据保留，下一PPO不开始。命中后在相关路径不调用resource_guard或final GPU。坏JSON/非object/缺未知action/read异常具名ControlProtocolError，原字节bin/base64/SHA和解析错误留证；publisher原子替换，监督器不再把用户control覆盖为failed。当前同一helper锁存语义正确，但F02是尚缺的跨异常native恢复边界。

## 已读CPU证据与诚实限制

48日志SHA全匹配；已读关键原文及报告中所有失败/更正说明。主要局部GREEN可见stdout：policy+manager41passed/11.67s；迁移+runtime18passed/24.57s；bounded runner/recovery5passed/67.14s；实际60 handoff两项2passed/43.92s；unproven duplicate1passed/8.26s；Q002整链17passed/81.70s；最后helper/window/Q01013passed/19.70s。后13项不等于把此前17整链全部重跑在最终SHA上；最终统一suite应承担同freeze验证。

报告保留早期bootstrap/夹具API/CaseTable属性/transition历史计数、manifest lambda root、Q010 V06Env拼写错误导致的失败，未将失败/skip当全过。raw日志多未包含统一command/EXIT尾注，工具exit归属采用owner具体回执，不伪造reviewer重新执行的exit；本轮reviewer仅两份新反例工具有直接actualexit0。

## Declined-to-judge／后续必须完成

1. 两项确认P2的实施修复；原owner保存原包/before/R1 TDD，reviewer只复审finding+新diff，不代写产品/tests，不重复同SHA通过项。
2. 两项修复后的最终同source/test/docs/observer freeze及完整CPU；当前v1虽据根回执219通过仍不是已修复闭包，也非acceptance。
3. 最终串行五GPU：D22 smoke8env2更新；R/L40 probe512+8/56、实际50诊断15原件和55→56；原源零学习512首二load及真实role/tile/四RNG/reset缓存/释放；45baseline；flat/H00各15/真实500窗。所有resource/费用需实际记录，不能由RecordingEnv或CPU JSON代替。
4. W&B实际发送/超时、GPU资源实测、完整Q008评价RNG隔离、训练效果/最终爬楼通过率/是否改进；本包没有足以作肯定效果判断的数据。
5. 最终manifest/独立MDJSON签署、生产首4818、自动正式启动与Git发布由根处理，reviewer没有全局accepted或启动权限。旧原件及已过A/root限定结论不改。

## 核心源绑定与本批修改

|文件|实际SHA256|
|---|---|
|src/tasks/velocity/v07/runner.py|`04cf47f640485493da475a7f04c54623f998664271e118a0309dd1280f2e93df`|
|src/tasks/velocity/v07/runtime.py|`20092bd3c26e63c4b15c2bf6aa6f7b26f06cb752a7992dcff907cd3e63c34a81`|
|src/tasks/velocity/v07/recovery.py|`b00ddd5601d641798895bb0f695e2a3116e2c5b8cb7536e7d7c7ec4d324e5868`|
|src/tasks/velocity/v07/curriculum.py|`a744970c2f0e3bd044c4dbb6acd54424c65098aebe6a201f86b2faf91202126d`|
|src/tasks/velocity/v07/operator_control.py|`9f752f6e5744a354c6fb01bbf3237ebc6e36192df2a2b65eda326e8d7169848b`|
|src/tasks/velocity/v07/window_evidence.py|`665382b02df90f0ca6f436e8990f9a3eb0b876a1e31cc10dfa5bd2ba6f49b533`|
|src/tasks/velocity/v07/continuation_v07.py|`fd4ed219e499ce3fd58ee5b66471649994c94063148589aa395dc7386a9b8b51`|
|src/tasks/velocity/v07/regression_policy.py|`c0ae8ca976501d4050f9377222b47f743c4d88c9d0c923dddee6cdaef208c8e4`|

其他28源/tests和48日志完整map见固定TASK_B_HANDOFF.json；TASK_B_REPORT.md实际SHA `c6caa0a5a68bb8de66d5f49d702ae13fec0989b987e4155d5c88d351b9e009c0`。本批只新增本TASK_B_CORE_REVIEW.md及/tmp反例回执；未修改36owner文件、执行者tests、旧C1原件或共享日志，未GPU/正式学习/Git写。根统一追加[logs/codex_changes.md](../../../codex_changes.md)，记录两个P2及真实限定CPU证据。修复后需新freeze，不把已有局部或v1CPU通过当豁免。


## B CORE R1限定复审：两个finding闭环

日期：2026-11-17T14:50:21.478504+00:00。完整读TASK_B_CORE_R1_BRIEF、TASK_B_REPORT的R1附尾、TASK_B_CORE_R1_FIX.diff全部245行/18173字节及新tests；只审原双finding和本轮diff新增破坏，没有扩大到其他逻辑。当前36文件、原36 before字节、两个原始RED/GREEN及metadata绑定均复核；精确5文件变更、31文件原字节不变，pure regression_policy SHA仍c0ae8ca976501d4050f9377222b47f743c4d88c9d0c923dddee6cdaef208c8e4。

### 每项结论与双verdict

|项目|R1结论|依据|
|---|---|---|
|F-B-CORE-01/P2|**ADDRESSED**|强制before state/policy摘要、消费次数与window数量、前后FIFO/EMA/generation和不可变原件相容，不能用空列表假装已证明消费|
|F-B-CORE-02/P2|**ADDRESSED**|正式resume_contract在已保存有效pause/stop锁存未解决时早拒绝，后续continue不能授权新worker或学习；无锁存工程异常仍可合法恢复|
|Spec compliance|**PASS_WITHIN_B_CORE_R1**|两项scope要求闭合，原正常对照保留|
|Code quality|**PASS_WITHIN_B_CORE_R1**|本轮diff无新增确认P0/P1/P2；当前B核心blocking=0|

本结论只覆盖本核心模块及R1闭环。原declined清单中的完整同freeze CPU/GPU、真实512/RNG/cache/资源/行为效果仍待验，不由57项局部或下述轻量复核替代。

### F01实现核对与合法对照

prepared现在保存实际before_state+before_policy摘要。applied另保存本次实际consumed_windows和independent_stop。validator先核前态摘要，再核数量必须等于window条目数；没有独立停止时按消费前实际FIFO/完成状态推出所需窗口数，每个存在窗口仍严格核实际文件/SHA、before前64、前EMA和measurement身份。未换代要求postFIFO精确为移除已证明64条后的剩余，并用原始行与原EMA系数审计解释postEMA；换代仅允许+1且原规则FIFO/EMA清空，同时旧代消费原件仍要存在。独立停止则要求确实已生效、前后课程/policy保持且无events/消费。

EMA复算仅验证缓存，未重新应用EMA/课程、未再PPO/评价、未回滚RNG、未增加每update全量保存。合法windows=[]（无消费或独立停止）、正常消费、合法generation reset、既有foreign-window拒绝均由owner受影响套件保留；新diff没有改课程公式/阈值、窗口排序或原writer。不存在用“删sort/吞异常”替代证据闭合的处理。

### F02早拒绝核对

root明确选择fail closed，R1只在recovery.resume_contract读取checkpoint已保存的terminal.evidence.control_error.latched_request；若action为pause或stop即RecoveryBlocked要求裁决。实际native runtime先preload(mode=resume)→resume_contract，随后才进入新worker model/env/pending/ready，所以该拒绝早于新GPU/env工作；不会读取后来continue消除旧请求。

mode=train仍可只读原异常证据，原error_type/stop_state=None/raw字节/异常文件和audit保留，没有伪造用户终态或增加自动pause恢复；没有有效锁存的坏JSON工程异常保留原合法恢复路径。owner真实runtime测试覆盖pause/stop×普通/pending四分支，在_snapshot IO异常与自动exception之后用新segment continue+resume_authorized=true，均在env构造前拒绝；源exception/control字节不变、无ready。合法无锁存坏JSON纠正后实际ready/learning guard通过。旧普通终态拒绝守卫未放宽。

### 已核实际RED/GREEN与独立轻量复验

已读原文关键失败/summary以及完整GREEN（含真实command和SHELL_EXIT_CODE=0）：

- RED：9failed、4合法对照passed、17deselected，121.69s，owner工具exit1；日志SHA `d00b4bad11d829464eeb6303f885457887baaa1cbf143904a0bc9ebdd2293900`。9项分别为5个缺消费proof负例和4个锁存跨native恢复负例。
- GREEN：五受影响文件**57passed、0failed/skipped/errors/deselected，331.61s，SHELL_EXIT_CODE=0**；SHA `2aab73ccf38c783a47a9c171695833ddf16e2728efe42339b0b2c66c8904678b`。仅既有jit弃用warning；没有重复跑该套件。

reviewer另实际完成7个轻量检查，最终工具exit0：正常64行同代消费接受；漏window拒绝；连beforeFIFO也漏存拒绝；即使重算before摘要并宣称0消费，postEMA0→1仍拒绝；pause、stop两种已保存锁存通过真实runtime.prepare入口在新worker model/env前拒绝，continue/resume_authorized=true不放行且无ready；无锁存坏JSON的native恢复合同合法接受。

回执：`/tmp/a2_v07_b_core_r1_independent_recheck.json`，SHA `1a3ad24e41db9d9d9adef27b88181340606edafc0a3d499c9701717918683b12`。F01使用真实manager/规范64行/当前validator；F02使用真实裸model/full非空Adam生成的schema9 checkpoint和受控已保存control_error audit，通过真实runtime入口，并将后续新worker model/env设为禁止调用来证明早拒绝。**本轮没有重新制造现场IO错误**，现场_snapshot失败与自动异常链以已读owner实际测试证据为准；不冒充重跑57项或GPU实测。没有PPO、物理仿真或GPU。

透明记录：轻量复核前两次尝试分别因reviewer临时夹具terminal=None写入audit的TypeError、临时terminal缺schema字段的ValueError而exit1；均为reviewer夹具错误，不是产品失败。最终改用真实runner生成完整terminal结构，不放宽生产守卫；前两次未当作通过，回执保留其原因，原tool stdout未另伪造持久日志。

### A2固定freeze v1→v2单行metadata补充

直接将当前validator与保留的原SHA b03afac67df40b16f574b6c8576497cd7f403a66eaa5f9ccb0a4f47e4fc9574e原件字节替换对拍，确认只有FREEZE_PATH从freeze/v1/V07_SOURCE_FREEZE.json变为freeze/v2/V07_SOURCE_FREEZE.json。当前SHA `55158db282850b881fb46fc8890b58b8e41338510b074a0e8e616099eac61ad5`；diff SHA `06aa98a6ed890229f317b7761dec562be680f1db0e10e36a3f9338625325944b`，VERIFY.json SHA `9f060b99f111e8352c4e3e630544824f4e49b95393869c12746ab16f8a1b9893`。

consumer默认仅读新固定FREEZE_PATH，没有v1不存在时回退到旧freeze的接线；wrapper也从同一validator取FREEZE_PATH。默认GPU输出标签v1未更改是目录命名而非source freeze复用，execute仍拒绝已存在输出目录；最终实际工程目录/新freeze由根绑定。此metadata delta的spec/code quality均**PASS_WITHIN_FREEZE_PATH_DELTA**；不重开A2其他实现范围，不改变其原限定通过结论。

### 当前固定SHA与继续待验

- TASK_B_CORE_R1_HANDOFF.json：`7e77f33f337ecf58a582b04eac38b804db3797b1a7528402ffa493ee57360ea4`。
- TASK_B_CORE_R1_FIX.diff：`ac1f1aee7c652bbe55e1ab3b11bdbed161392377074f3db9e6a5119b8695fcaf`。
- B_CORE_R1_BEFORE/MANIFEST.json：`c3ac06445213bc52b5656caec14d8a510debba65960930be8a1d6db6dcbef699`。
- TASK_B_REPORT.md：`ede690a056b7b3597f036d0fa06cd60263996bbc810502cdafdee970d3559d48`。
- 当前runner.py：`537a1cd850b09e536bf3454ef3c6da4d945b35a901d2cf33356af9cdba4f4d00`；当前recovery.py：`65b10f1a27c0369aa63924eb274fde9f27dc563da121fd0ff71db80f8fc9e695`。

根已报告新v2冻结source829683d61d7732cf683c96ebc6f659769e60c7324a386ba95227ac3fb53e274a、contractb8857d4b18482f391eb9ec713ea8b6a7236af77bb14d7f8fdd6bf9f931fbba77，94source/22native tests/30docs/3root inputs；本轮只绑定上述实际R1文件与delta，**未把在途完整v2CPU或尚未执行GPU当作已通过**。最终仍需统一实际证据后独立MD/JSON放行。

本批只保留历史并追加TASK_B_CORE_REVIEW.md和写/tmp独立回执；36owner文件与A2当前validator在复核前后不变，未修改执行者tests、共享日志或Git。root统一追加[logs/codex_changes.md](../../../codex_changes.md)，记录本批两项ADDRESSED、实际轻量结果及reviewer夹具失败透明说明。
