# 旧 447：S1.0 封存证据与依赖连续性独立验收

日期：2026-11-20。**验收通过：旧 447 的五套原教师基线、600 病例校准、finalize 索引及已记录依赖/资产身份连续性。** 六项义务均已关闭；正式学习操作和样本均为零。本报告不披露 final 成绩、每组成绩、行为指标值、轨迹表现或选模建议。

此结论完成 Q010 Task 0 所等待的旧端完整性验收。后续可按既有规划开展限定修复与新快照准备；旧→新证据复用、资源执行和学习仍分别需要新源码比较、直接绑定清单及对应独立准入。本报告没有签发这些批准。

## 1. 执行身份和范围

- 项目：`/home/xxs/dog/unitree_rl_mjlab_a2`。
- run：`a2-v09-s1-seed42-r3`；目录 `logs/a2_v09/s1_seed42_r3`。
- SDS SHA：`500ad32aaa946c927459427f3a65565b15d4b77fca4be214817121d15c742976`。
- 生产源码 SHA：`44780cd5d6f9ee4029fb38b2f9348a872268e7bedf995d4b36e2e30ec2f2163f`；冻结清单 `source_freeze_s1_q009_final.json`，322 文件。对应实现提交为 `c5b2878cc747ef675e0a637a8146138d69cd8e7a`，后续文档提交不替代源码字节身份。
- 唯一原 6400 SHA：`30ed23ba16f7e490afca001fbbecadb4a11ff4b11733a5ae7ac7d0c841f9a8b2`。
- 完整预留计划 SHA：`fa439aac08c47fda060413cbae2942ce92ef39194d49bf850dd68a9f5ea7b418`。

审核员等待各套 summary 存在且对应 obligation closed 后，使用 CPU、禁用 CUDA 逐套验算。final 数据只按独立 `freeze_audit` 完整性范围读取；没有进行正式解封、行为门禁判断或模型选择。审核员没有执行物理仿真，没有修改源码、权重、病例、阈值、原证据、外部账本、A/B 独审或原 S1.0 签字。

## 2. 覆盖、计费和关闭

| 义务 | 固定病例覆盖 | 批数/并行槽 | 实际控制样本（含 padding） | 关闭 sequence | 独立完整性 |
| --- | ---: | --- | ---: | ---: | --- |
| teacher_gate_A | 75×20=1500 | 75 / 20 | 817,400 | 152 | 通过，原单项报告保留 |
| teacher_gate_B | 75×20=1500 | 75 / 20 | 869,800 | 303 | 通过，原单项报告保留 |
| teacher_final_1 | 75×100=7500 | 75 / 100 | 4,760,300 | 454 | 通过，成绩封存 |
| teacher_final_2 | 75×100=7500 | 75 / 100 | 4,327,600 | 605 | 通过，成绩封存 |
| teacher_final_3 | 75×100=7500 | 75 / 100 | 4,644,400 | 756 | 通过，成绩封存 |
| calibration | 75×8=600 | 75 / 8 | 311,152 | 907 | 通过 |

五套教师共 25,500 例；另有独立校准 600 例。合计实际计费 **15,730,652 个环境控制样本**、**62,922,608 个物理子步**。控制周期为 0.02 s、每周期4个0.005 s物理子步，既有动力学和计费口径未改。上述费用保留在原 S1.0 账本，不因后续复用而重写、退费或再次计费。

每批费用从原始首终态独立重建为 `并行槽数×本批最后一个首终态步数`，与 reserve/complete、console 及账本相等；提前结束槽的后续物理等待步全部计费。六项 completed_cases 均完整覆盖各自 manifest，reserved=0、closed=true，绑定唯一6400，未发现 unknown/interrupted 结算或工程无效证据。

## 3. 原始证据的全量检查

五套教师全部 **375 PT、375 geometry JSON、25,500 条首终态**已检查；校准另有75个values PT及75个geometry JSON。final三套分别完成75批全量核对，未按成绩抽样或过滤。

- 病例 SHA 唯一、完整等于对应冻结 manifest，批次顺序、slot、case_id、streams、初始姿态及生产者身份一致。原始轨迹 step 从1连续到terminal_step，长度对应 control_samples/end_control_step 且不越原 hard timeout。
- PT 内 geometry 与独立 JSON 统一元组/列表表示后内容一致，certificate SHA 全部重算通过。原23地块表一致；按证书中保存的 compiled 位置/尺寸重建只读描述对象，原 `index_geometry` 检查每批231个地形box和 tile/step/top/support_bounds 一致。本次没有重建物理环境。
- 全部教师轨迹 metrics 使用冻结实现从原始数据重算，与 episodes 一致；平地速度 RMSE、接触滑移另按独立公式核算。张量有限，nonfinite/contact_overflow mask 为false，无 engineering_error 文件。
- 保存的固定初始教师观测为3×48×32地图和5×45历史；CPU重新载入唯一6400，逐批 action/value 与已保存输出满足 `rtol=1e-4, atol=1e-5`。这里核的是已保存初始观测，不声称重推全轨迹每一时刻的完整教师输入。
- 补充全25,500例原始时序检查：`previous_action[1:]==action[:-1]` 精确成立，首 previous_action 为零；actual sample_command 与冻结case及Q002 actor索引对应，is_cruise序列精确一致。指令浮点核算使用 `atol=1e-7, rtol=0`。

上述浮点比较公差用于核算CPU/GPU或float32算术一致性，未改变任何SDS行为阈值。final记录中的结果字段仅在独立工具内部检查汇总与原记录一致；没有导出成绩或行为分析。

## 4. 独立校准的样本分母与 FP64 核算

600个校准病例完整等于 `train_calibration` 冻结manifest，SHA为 `5a2f1bb9dd934abde48105dd3b133c3fcdc660be90c7444bc086116d60939f7a`。75批values PT的 source checkpoint、病例/slot、记录的初始姿态与streams、geometry和账本均通过。

逐批按每槽首终态步数重建每一控制步的 active slot 列表，原 `case_slots` 与重建序列精确相等；values为对应的有限 `[n,1]` 数组。有效原value总数为 **289,564**，恰等于600例首回合控制步数之和。实际计费311,152，其中 **21,588 个等待样本未进入value分母**；没有以零、重置后状态或完成槽后续value补齐分母，也没有按成功筛选。

用全部原values转FP64一次性核算总体统计：

`mean=ΣV/n`，`M2=Σ(V−mean)²`，`sV=max(1, sqrt(M2/n))`，不使用样本标准差的 `n−1` 分母。

- FP64 mean：`4.4775978392048605`。
- FP64 M2：`1321803.1406824554`。
- 原生产者文件冻结 sV：**`2.136540383972894`**。
- 独立重算 sV：`2.1365403839728914`，在 `rtol=1e-12, atol=1e-10` 内一致。

后续应使用原 `calibration.json` 的冻结数值；独立重算只用于验证，不替换该参数。类型、有限性、方法 `population_std_max_1`、600病例、样本数与物理费用均通过；实际311,152≤1,000,000。原完整覆盖target945,800仍是预留义务上界，不要求为了凑上界追加物理样本。此次校准核验围绕原values/slots、首终态、冻结生产代码和总体统计，不声称已保存数据足以逐时刻重推完整critic观测。

## 5. finalize 索引及外部账本

已等到 `commands/s1_0_serial_complete.json` 完成收据，核对其中baseline SHA、旧code SHA及学习0，再读取最终索引，避免读半写文件。

`baseline_admission.json` 的 schema=11、baseline_complete=true；五套teacher_cache的绝对路径、case_count、checkpoint/code/manifest身份、episodes与summary内容SHA，均与原文件和已通过的逐套独审证明对应。校准绝对路径、原文件SHA、manifest和sV一致。全部逐文件证明所引用的原始文件再次hash，内容未变化。resource_admitted=false、behavior_restored=false保持明确；没有正式学习操作或样本。

账本至校准关闭的sequence907全部SHA链通过，事件SHA为 `503fbb81eb11acff4bc068d722c4cba65905c8a1424359c9bf3d01a539ee737f`；原始闭合前缀文件SHA为 `c526987c278867f390ae7c02dd50ccd5de898f8df734ed0d6a157f7441e4194e`。索引及校准所引用的外部账本前缀身份有效，没有要求计数退回旧值。

`calibration.json` 内嵌ledger是保存时sequence906的历史快照，当时calibration义务尚未执行close；原文件内容保持原样。实际close为907，finalize索引引用907。不能将内嵌历史`closed=false`误读为当前未结账，也不能据此回退外部账本。

## 6. 旧端源码、依赖及资产连续性

finalize后重新计算 `spec.code_snapshot()`，完整文件集合与全部322字节摘要仍严格等于旧447。原源码完整归档 `source_archive_44780.tar.gz` 已独立验算SHA `f62f12d5e9513072c2d315d06fa05bb637a82a856515af7c78118a8410b00d28`，tar内322文件与冻结清单逐项相等。

此前本机身份清单记录的 **34,883个唯一常规文件路径**全部重新hash，零内容变化、零读取错误；八关键包非缓存文件集合、全部stdlib/site-packages非缓存Python源集合、Python启动 `.pth/egg-link` 集合均无新增/删除。

- 主清单覆盖 A2全部资产（含XML引用的17个STL）、mjlab/mujoco/mujoco_warp/warp/torch/numpy/PIL/imageio_ffmpeg八关键包、已安装分发元数据、Python native extension、原实际进程映射常规库和执行绑定。
- Python补充清单覆盖stdlib/site-packages全部非缓存.py/.pyi。实际final_2进程PID3072253的252个mapped常规文件另做过只读抽查，全部已被主清单覆盖，零新增、零内容变化。
- 启动补充清单记录4个相关路径；唯一editable工程指向本a2仓库。该补充是在旧final_3运行期间采样，不伪称其中此前未覆盖的.pth文件已在第一份清单中被测量。

覆盖边界保留：这不是live进程sys.modules完整转储，也不声称覆盖所有可能晚加载的非Python资源。主清单的7条`/dev/zero (deleted)`匿名设备映射和3个NVIDIA字符设备不属于可普通文件hash的代码库；不以它们证明驱动内核状态。运行生成pyc/cache按原规则排除。Q010仍须对新端实际依赖/资产与旧端做逐项匹配，不能把本报告当作通配兼容批准。

## 7. 封存证明身份

以下路径均相对于run目录。本机JSON不提交Git，final原文继续封存。

| 文件 | SHA256 |
| --- | --- |
| `teacher_final_1_independent_integrity_20261120.json` | `8019910a3291b5214be30921f4706599b237de51eebcb9fce534a41f8cb27573` |
| `teacher_final_2_independent_integrity_20261120.json` | `ca2638ab75f0b5e0d2024b48df082875f8233c71d22c9d04b69e77c48f758047` |
| `teacher_final_3_independent_integrity_20261120.json` | `9a942eefd7594395acf5ed0b8e6fcf003c961c5f7761651c855124a8a5907e36` |
| `calibration_independent_integrity_20261120.json` | `e09cc2591cb55fa4030bd3d11bb32effed3390def34424fdc8164bc0b387b37b` |
| `s1_0_raw_temporal_independent_integrity_20261120.json` | `12cb864f4a1795748e3155835f1639d39b97e24882622ecf84ba005580039273` |
| `s1_0_finalized_independent_integrity_20261120.json` | `e4117a495b0abdac58066a0d0ad2a493bfb034894666384abe087e78e48321d6` |
| `baseline_admission.json` | `ffa28f8b9f78b7c59379e4168b7252844bd697955d38048b688046dc83020ae2` |
| `calibration/calibration/calibration.json` | `9ea9e55dfc66e4a140a99cee87171ea805c47ccc7b310a8b1928342b26d9fc6e` |
| `commands/s1_0_serial_complete.json` | `c2d40f968af06d0ebc279b2929f5b05da19cf80dda06ee74844b1998df25faec` |
| `old_runtime_identity_447_20261120.json` | `2375af6769d864a586cbe85396f9cfea89c614a542b12358b7e90534f774c7c5` |
| `old_runtime_python_sources_447_20261120.json` | `a3f4c4bf410b6196407190cec1d453e6afb2d0d3e973d69d293b4b6fe7544400` |
| `old447_python_startup_identity_20261120.json` | `bbe47d8dd159ab9a65ed3ca0f4ebc6a52baea126d1e87686b05277f8d9d40791` |

A/B独审报告和原S1.0报告/批准保持原样。新增报告及本机证明的修改日志由主agent追加至 `logs/codex_changes.md`。

## 8. 后续范围

旧端完整性验收完成。此后严格保留原447证据的真实生产者与内容SHA，不重写为新版本。final原文的独立完整性检查和hash验证不构成正式解封；后续实际消费仍须fresh `validate_final_unseal`。

资源入口 R-RES-ENTRY-01、Q010直接兼容清单、新源码比较、资源工程批准及真实探测、Q008初始/后续证据和S1.1/S1.2学习均不由本报告批准。当前没有v0.9学生行为改善或恢复结论，未授权S2/S3。
