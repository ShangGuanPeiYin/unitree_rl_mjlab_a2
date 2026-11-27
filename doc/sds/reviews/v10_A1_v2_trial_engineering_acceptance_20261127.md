# A1 v2资源与50轮试训护栏：最终工程验收（2026-11-27）

## 工程结论与实际运行边界

新工程CPU验收通过，最终source `d54265108c298d5920ae8ea9316fd3d48be4b75ce989ac0507bdbf9a2b91498e`。全部18个test_v10*.py：189 passed、5 skipped、1上游warning，116.26s；GPU/3GB专项仍跳过，旧项目全量收集冲突仍未整体通过。源码/文档diffcheck通过，原始测试stdout保留框架空格不为样式改写。

**没有新的资源/行为通过结论。** 2f v1资源失败及其98304费用不变；当前global probe3391488、余442368、正式学习0。新v2、112、50轮未启动，生产canonical trial guard未创建。当前只读60s可用RAM最低约10.056GiB，低于12GiB启动条件；不得以工程通过绕过它。

冻结SDS v1.1仍SHA `eb95905ed24bf0674f94a1b02dab02256ee11e40a20ed1cd96eb24daeb1b49a0`。新顺序及参数由独立文件v10_A1_trial50_resource_v2_ruling_20261127.md（SHA4ae096ea3e16e80daf6414d4976231f051873c27fbdfa3ec16383bd6e7d3248e）关联；旧已签裁定和2f合同/失败证据均不覆盖。

## worker试训上限前后

旧worker的stop-round50按confirmed物理样本判断目标，实际charged仍受原D1 24576000上限；崩溃uncertain或回退旧checkpoint后，不能据此保证试训只花2457600。

新显式trial模式仅D1/N1024/50/2457600，worker局部有效cap=min(原D1,trialcap)，每次启动完整rollout须charged剩余≥1024×48=49152，逐stepQ1账本仍唯一收费。canonical guard绑定trial-id和同一学习ledger，禁止换run/ledger/ID赠额。当前账本0；未来本次所有试训及恢复沿formal_A1_20261127_approved/learning_costs.jsonl。

checkpoint extra.trial中的retained_completed_updates跟随实际模型更新和checkpoint恢复；外账historical_updated单列，恢复较旧checkpoint不借历史梯度虚称当前模型含50更新。首次step前实存完整retained0 checkpoint，正常50轮跨原2457600保存周期并按原边界hardlink去重；初始/最终保存失败均抛错非零。旧initial恢复遇uncertain/已耗不足完整rollout，返回trial_budget_exhausted而不补采；50 retained更新完成才trial_limit。

试训始终gate_reached=false、teacher_admitted=false、accepted=null、training_accepted=false，不能被正常控制器误当课程门禁通过；trial checkpoint不带参数走默认无帽续跑被拒。正常非trial路径原行为不变。生产CLI沿已有gpu_execution_lock互斥，内部Python worker无额外自有进程锁（原结构），未冒称新增另一锁。

## 资源v2前后及明确限度

原full协议保持1fill+5warm+20measure；A1 v1的1+2+2不重写。因余额442368不足v1两栈491520，规划批准固定v2两栈各1fill+1warm+2measure=196608，共393216，最大global3784704余49152。这是显式关联短补证，不是重新20测量长验；任栈失败停止，不凑余量/换合同重测。

父CLI以纯stdlib/importlib路径加载，在任何src/Torch导入及sequence/claim前须连续60s每点MemAvailable≥12GiB；未达返回0 physical/无claim。child只核原父窗口与claim绑定、owner存活，填库前即时≥9.5GiB，不额外要求第二次12GiB/60s或让已开始序列因窗口年龄失效。运行原4GiB门槛和GPU阈值保持，满196608 FP32 replay、Adam、真实save/CRC/SHA/free/load及RNG/下一采样等价/全程峰值均保留。不kill/调整他人进程，不清峰值。

v2新合同绑定旧b48两完整通过证据、2f失败仅context=false、新source/差异、独立审核和新裁定。300源码逐文件逆A1重建旧实际b48精确一致：13已有修改、16新增、271保留实际旧字节，未夹带既有dirty。42核心证据中40项byte同、env精确记录节点AST等价、runtime仅worker显式trial差异，实际probe使用的其他函数及模块AST同；没有把env/runtime全文件假称byte相同。

v2全协议唯一claim跨contract/seq/outdir且后续传False也不能逃逸；parent RAM不达在claim前不消耗物理机会，child已claim后失败不自动重试。同cap承继严格父SHA/id/ledger/SDS，新增锁内退休复验覆盖begin/assert/acquire/claim/reserve/validate，修复“锁外预读旧授权后，继任退休旧授权仍可begin”的实证TOCTOU。历史费用不退、不双计、不重置。

## 独立证据与执行顺序

| 项目 | 证据 |
|---|---|
| worker guard与50边界 | trial_guard10项；默认runtime回归18项；另一代理独立trial+logging+protocol29passed，v10_A1_trial_guard_independent_20261127.md SHAaa292eb878e33237e41211a121c9c553a79a8ba94526c3b66eb051ccb438ab94 |
| v2/无src启动门禁/退休竞态/唯一claim | 实现定向87passed；独立22passed，v10_A1_v2_resource_independent_20261127.md SHA19e4bd4534675439b95b39c34e823a77cf2c1eb049240124366cbbc9b8501f18 |
| 最终整合 | A1_v2_engineering_20261127/final_v10_cpu.log全部18文件189passed/5skip；编译/CLI help/diffcheck |
| 旧实际source对应 | actual_baseline_reconstruction.json SHA2675a08a4b799d1b313e7af1d73d9037e2a498a9980b7792b44b5a1bb1279d9d |
| 核心不变/全差异 | invariants.json SHAf4ab64229f2e0633cf0c9de6f2ccd1f0dada8ef271434dd0c9519787c4180811；source_diff.json SHAcdd88c3e60295fd2e0c646d98a0c242181d57f53d013d674bf0104d3aa816646 |

原A1纠偏公式/增益、教师权重、学生请求/history/真实动作、奖励/损失/LR/随机化/地图/路线不改。下一步严格：RAM门禁→v2两栈→固定112例28组每组4/4→D1本次试训最多50完整更新（每轮48步、1024环境）→保存停报。全部计入原D1/总学习额度，完整教师A/B只后移且保持未通过，最终验收不取消，不自动继续课程/PPO。loss/训练统计不是独立行为验收。
