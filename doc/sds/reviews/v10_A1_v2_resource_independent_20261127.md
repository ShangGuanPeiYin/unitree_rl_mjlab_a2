# A1资源补证v2、启动内存与授权退休独立工程审核（2026-11-27）

结论：对最终冻结源码 `d54265108c298d5920ae8ea9316fd3d48be4b75ce989ac0507bdbf9a2b91498e`，资源v2、父子启动内存门槛、费用承继与唯一尝试的工程审核通过，未发现剩余阻断。本审核发现并实际复现了一处退休授权的预检/事务竞态，实现方修复后，审核者独立CPU回归通过22项。此结论不是资源v2实测通过，不是112例教师诊断、完整教师A/B或试训行为验收。本文不签发资源通过证书，不修改受审源码，不增加物理样本。

## 范围、角色与冻结身份

本次审核对象是 `resource_a1.py`、`resources.py`、`probe_budget.py`、新增纯标准库 `headroom.py`、CLI和调度器的v2接线，以及相应CPU测试。审核者没有实现这些资源增量，只读复核实现与证据，并新增本报告。

试训guard由本审核者实现，**不在本文自行批准**。该部分引用另一位审核者的 [试训护栏独立报告](v10_A1_trial_guard_independent_20261127.md)，SHA256 `aa292eb878e33237e41211a121c9c553a79a8ba94526c3b66eb051ccb438ab94`，其独立29项CPU通过。本次确认其受审 `runtime.py` SHA仍为 `00677d34add3c195061d4167b0eff2611ffa8007a58c07556b4d0fecbae7f043`；新的CLI三项trial参数及原worker参数传递保留，CLI仅另加v2解析/启动检查和对应分支。试训完整保存/恢复的独立结论来自该报告，不能归于本文的自审。

本次绑定新增裁定 [试训50轮与资源v2裁定](v10_A1_trial50_resource_v2_ruling_20261127.md)，SHA256 `4ae096ea3e16e80daf6414d4976231f051873c27fbdfa3ec16383bd6e7d3248e`；冻结SDS v1.1 SHA仍为 `eb95905ed24bf0674f94a1b02dab02256ee11e40a20ed1cd96eb24daeb1b49a0`。没有覆盖旧SDS、旧通过或失败证据。

## 核查结果

| 核查项 | 实际实现与审核结论 |
|---|---|
| 固定短程补证 | v2固定N1024，每栈1填库+1预热+2测量，共4×1024×48=196608样本；DAgger和PPO合393216。每栈完成学习更新147456样本。明确不是5预热+20测量完整复验。 |
| 保留旧协议 | 原SDS full入口仍1填库+5预热+20测量；v1仍1填库+2预热+2测量。v2用独立protocol、契约与CLI，不能以参数把短程冒充full。 |
| 真执行体与完整状态 | 两栈仍调用共享真实执行体；六任务各32768、合196608条FP32 replay，真实Adam、完整保存、流式SHA/CRC、释放旧replay、加载、模型/优化器/全局及replay RNG/下一批等价检查保留。新增仅在v2填库前加入内存拒绝分支。此次没有重跑3GB测试或GPU来重新证明其资源效果。 |
| 父启动内存 | 生产CLI先用文件路径载入纯stdlib helper，连续60秒逐样本MemAvailable≥12GiB；遇任一下跌或读数失败立即拒绝，不重置计时继续等。检查先于src.tasks/Torch导入、协议claim和物理采样。真实CLI CPU反例验证未导入src.tasks/Torch且未创建费用账本。 |
| child内存与原窗口 | child绑定原父报告SHA、路径、owner和活跃PID；不另做12GiB窗口，也不要求第二栈仍处于父报告的fresh时间窗。填库前单次要求≥9.5GiB，低于门槛则collector不执行、费用0。归档证书验证不要求历史父PID仍活跃、不将旧报告按当前时间失效。 |
| 全程原硬门槛 | GPU总占用≤7680MiB、任务≤7168MiB、MemAvailable≥4GiB原条件保持；ResourceMonitor贯穿执行/保存/释放/恢复，阶段采样和进程RSS/HWM合并，未清零峰值。12/9.5GiB只是前置条件，不能保证背景负载以后不变化。没有kill或调整其他进程的实现。 |
| 唯一尝试 | v2持久exclusive protocol claim；首栈失败即停止，无N回退。已存在exclusive记录时，即使改变contract、sequence/outdir或后续caller传False也不能再次claim。child还必须匹配原sequence、contract与输出位置；原v1默认规则保留。 |
| 费用与容量 | parent在claim/child前申请393216完整容量，child每栈196608独占lease；lease不是实际扣费。物理仍逐步reserve/mark，uncertain保守charged、不退费；zero-sampling失败不产生费用，失败claim不会赠送重试。未决事实不靠释放容量退款。 |
| 同上限承继 | 新旧SHA/id/cap/ledger/SDS/scope必须匹配承继关系，父授权须已registered；继任注册持久退休父SHA。全局active互斥与累计cap维持。退休检查修至每个授权驱动的实际锁内事务，不能使用早已读到的旧授权快照重开。 |
| old通过与failed对照 | 正资源证据只引用旧实际b48来源的1024两栈full通过文件；2f v1失败文件用固定SHA绑定且passed=false、positive_resource_evidence=false，仅作失败对照。没有把失败变为正资源证据。 |
| 契约与证书 | 新契约绑定v2裁定、当前完整source、完整source_diff、物理不变证据和独立报告。关联证书逐字段重建，绑定两份真实raw result、精确成本/更新数、阶段记录、父/child内存证据、旧full与失败对照；篡改run或漏差异会拒绝。 |
| 后续入口 | diagnostic资源验证仍先检查当前code/SDS/N再完整verify_linkage_selection；调度器显式识别v2的1/2轮计数并验证关联证书，不仅依赖passed布尔。资源失败不能以此启动112或试训；允许顺序以新增裁定为准。 |

## 本次发现、修复与独立回归

初审发现：`_authorization()`在独立锁内检查旧SHA尚未退休，返回后，`begin_sequence()`再取得写事务锁。若两次锁之间继任授权完成注册、退休旧SHA并结束序列，已注册的旧授权会绕过原register分支的退休检查，再开启旧序列。

审核者在临时CPU账本实际执行确定性调度：注册old并finish→取得old的合法授权快照→同cap/same-rulings successor注册并finish→让old从先前成功预检恢复。修前输出为 `retired_old=true, stale_old_begin_succeeded=true, physical_samples=0`。这是一次实际工程缺口，未涉及真实全局账本或物理采样。

修复后的共用 `_assert_current_authorization` 在实际事务锁内校验退休集合；register/begin、assert、acquire、claim、reserve及validate入口均覆盖。reserve的检查先于旧uid幂等返回，不能靠重复uid继续使用退休授权；mark已发生事实等路径没有被错误改成事后拒记。确定性六入口回归验证拒绝时账本字节不变。另补持久exclusive_protocol，封闭换contract或降低caller flag的再次claim。

审核者独立执行的最终命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/xxs/anaconda3/envs/a2/bin/python -m pytest -q \
  tests/test_v10_headroom.py tests/test_v10_resource_v2.py \
  tests/test_v10_resource_a1.py::test_a1_protocol_counts_are_separate_from_full_probe \
  tests/test_v10_resource_a1.py::test_real_invariants_keep_env_byte_difference_and_exact_recording_ast \
  tests/test_v10_probe_budget.py::test_retired_parent_cannot_reopen_after_successor_finished_same_rulings \
  tests/test_v10_probe_budget.py::test_retired_authorization_stale_precheck_rejected_inside_transaction \
  tests/test_v10_probe_budget.py::test_exclusive_protocol_attempt_blocks_changed_contract_even_if_flag_omitted
```

实际结果：**22 passed，1个上游torch.jit.script弃用warning，19.98秒**。此前单独headroom三项独立执行为3 passed/0.09秒，属于重复子集，不累加为25项。CPU测试中的时钟/环境/child结果为明确工程fixture；实际共享stack的prefill失败路径、真实CPU全局账本事务、模板/AST/source差异重算和证书验证确实执行，不能冒称真实1024物理或资源通过。

另外回读实现方原始日志 `logs/a2_v10/A1_v2_engineering_20261127/v10_v2_frozen_green.log`，记录五文件87 passed/53.93秒；退休RED记录6项竞态失败及1项exclusive缺接口失败，GREEN7 passed/12.43秒。本报告的独立通过数只采用审核者自己运行的22项。主代理最终18个v10测试文件统一CPU回归已完成；审核者实际回读 `logs/a2_v10/A1_v2_engineering_20261127/final_v10_cpu.log`，结果为 **189 passed、5 skipped、1个上游warning、116.26秒**。GPU/full-replay opt-in未启用；跳过项目不算通过。该189项是主代理执行并由审核者回读的整套结果，不能冒称审核者另跑了整套。

## 源码不变性与三份最终证据

审核者按统一identity算法独立重算最终source为d542…，与主代理冻结值一致。另逐文件读取actual-baseline证据：13个A1 modified从07fe旧blob恢复，16个A1 added排除，271份实际旧字节保留，逐个校验old/new文件SHA，独立重建准确得到 `b48e3fd1601ff2926272ed014d846fd480fe24a8c56d77100d646f933b92991f`。四份原dirty（scripts/play.py、scripts/train.py、v09/s2_train.py、test_v09_s2_1_train.py）未被错误归为本次A1修改；source_diff中旧有v02–v05等差异也不意味着本次新增。

物理/训练不变证据明确 `unchanged=false`、`invariants_satisfied=true`：env整文件与旧源不同，只允许精确批准的recording语句AST例外；runtime整文件与旧源不同，只允许worker及惰性None新签名的例外，其余模块AST和实际probe collect/save/helpers路径完全一致。检查还拒绝module执行副作用、collect/save变化和worker执行型返回注解。本文没有声称runtime整文件字节相同；其worker试训分支采用前述另一审核者独立证据。

最终三证据均已实际回读并独立算SHA：

| 文件（相对logs/a2_v10/A1_v2_engineering_20261127） | SHA256 |
|---|---|
| actual_baseline_reconstruction.json | 2675a08a4b799d1b313e7af1d73d9037e2a498a9980b7792b44b5a1bb1279d9d |
| resource_evidence/invariants.json | f4ab64229f2e0633cf0c9de6f2ccd1f0dada8ef271434dd0c9519787c4180811 |
| resource_evidence/source_diff.json | cdd88c3e60295fd2e0c646d98a0c242181d57f53d013d674bf0104d3aa816646 |

## 受审文件SHA256

| 文件 | SHA256 |
|---|---|
| src/tasks/velocity/v10/resource_a1.py | 0a6a7a0eab96a159f19fca451d1549efb9f3416a16c6dc6efd2b3f01886a6c22 |
| src/tasks/velocity/v10/resources.py | a3a9d51aa1b1d5b0330d7eb2ce31eef7467b83ef1fa79f71369ea7ed3c1b65f0 |
| src/tasks/velocity/v10/probe_budget.py | f3f08c6d26df80f5acd47b463e6c4de2b01bd918e7372d309988219e6973f780 |
| src/tasks/velocity/v10/headroom.py | 200387366491f4d1d16a035db9ea231f864f489dbf66f4456e263dc2f62eed80 |
| src/tasks/velocity/v10/orchestration.py | 23011556f52da8816b7d441ab04594b146cf065cd9b1a2752586f7442b2b76e3 |
| src/tasks/velocity/v10/diagnostic_a1.py | 9fb5c774251535893f6d33300209f1ab75238e68d474fb2a9d4f3143ca252d72 |
| scripts/run_a2_v10.py | 1c5a3f5f0254581d27a91bea3938ef3b94d88a542d86dc071c5cafa0bfe6dc20 |
| tests/test_v10_resource_v2.py | bb5bdf1be448eee86073c6cd480e0a2c56e42a0fc191dacb19779a8cc60448c1 |
| tests/test_v10_headroom.py | 0b76e1eefd7059df056a4ad5148e29058bbd5c8e1d0bf65f0dfe85839f6ffc40 |
| tests/test_v10_probe_budget.py | 3846f16bd10eec0be19a5f3ecef9b31c6294c1d7cdd54e0a3184ef795dd5234a |
| tests/test_v10_resource_a1.py | 2dc45d40faccadec5bf9d533dc4fd939daf5851426f4d72fee4291d8e113d3d7 |

## 费用与未验证边界

物理前快照 `logs/a2_v10/A1_v2_engineering_20261127/pre_physics_global_costs.json` SHA256 `62407fe480ea95b1b9b61217506520f5a7d71e87869c2051b21bdd674c73e1d1`：probe charged=executed3391488、updated3170304、uncertain0、正式学习0、本轮工程新增物理0。原probe cap3833856不变、余额442368；若未来v2两栈全成功，计划增393216至3784704、剩49152，**这是计划算术，不是已发生费用**。

新RAM前置门槛尚未取得真实通过记录，v2/112/50均未启动。本文没有执行或通过连续60秒真实机器准入、9.5GiB真实child准入、GPU资源v2、3GB满库再测、112例诊断、完整教师A/B、50轮物理试训或后续课程/PPO；不声称资源改善、教师侧漂改善或行为能力通过。只有显式批准当前source关联契约、同cap授权承继和既有门禁满足后，主代理才可按裁定次序执行；任资源/112失败仍停止，不能靠本文绕过。共享修改日志、最终工程验收、提交与推送由主代理统一办理。


## 新同cap授权元数据只读核对

实际回读 `logs/a2_v10/probe_authorization_A1_v2_inherited_20261127.json`，SHA256 `37e3a3c038abb1213433677f6a6f68c8ece098b0fe86d352c78606bac215bdee`。父文件为 `probe_authorization_A1_inherited_20261127.json`，父SHA `49760a42facac7017b88db514b3beb09f8c7fb9b75458ac932a3a7bcc86d2d6c`，父id为user-A1-inherited-20261127-cap3833856，新id为user-A1-v2-inherited-20261127-cap3833856。逐字段确认同3833856 cap、同全局ledger、同SDS、同probe_only scope、同基础rulings SHA；新增amendment路径/SHA与本文独立裁定一致，approval_source保留原用户数字确认与规划直接裁定来源，不新造费用池。prior3391488、remaining442368、下一v2最多393216算术一致。

仅以普通文件读取核对全局journal中的已注册授权SHA：父49760…存在，新37e3…尚不存在；未调用begin/register/claim/物理入口，未写全局账本。生产canonical trial guard仍不存在。新文件的元数据一致不表示已经注册、退休已经发生或物理已经获资源准入；真正继任注册时由已审核的同锁检查及持久退休事件执行，新旧不能并行占用。以上只读结论不扩原cap、不退款，也不替代真实RAM/资源/112门禁。
