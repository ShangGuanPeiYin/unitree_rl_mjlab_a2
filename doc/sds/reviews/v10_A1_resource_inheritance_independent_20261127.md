# A1资源关联与同cap授权承继独立审核（2026-11-27）

## 结论与审核边界

审核人：`/root/learning_stack`。**A1新增资源关联、授权承继、一次性执行及阶段容量接线的CPU工程逻辑通过，当前无未解决阻断。** 根仍须使用最终统一CPU回归的实际结果；本报告不代表A1新GPU短测、教师能力诊断、正式准入或训练已通过。

本报告批准的是其他实现者本轮新增/修改的 `resource_a1.py`、`probe_budget.py`、`resources.py`协议参数化增量、`orchestration.py`和两CLI接线。审核人曾实现原公共资源/Q4生命周期core；不将其重新自批，原core依据是既有独立审核与下面固定SHA的旧完整双栈证据。审核人本轮实现的 `teachers.py` A1公式不在本报告的独立批准范围，须由另一审核者审核。

最终绑定源码指纹：

`d90f857d6b28907c2c560dfa2b30d8bdc3e16660df91fe518c8cf92046286065`

中间源码 `8daf0658…` 与其146项CPU结果只保留为历史，不作为此次最终源的替代证据。此次没有新物理控制样本或GPU探测。

## 精确证据关联

路径基于项目根目录。

| 证据 | SHA256 |
| --- | --- |
| `logs/a2_v10/A1_engineering_20261127/actual_baseline_reconstruction.json` | `92f4e3d23b168a8dc89a2346a787ec2b293c0d7002a0443d22306b7a76445bd6` |
| `logs/a2_v10/A1_engineering_20261127/resource_evidence/invariants.json` | `07015705ca1d8951393cbf42597fde306247ec534b498408046084fae22c75f4` |
| `logs/a2_v10/A1_engineering_20261127/resource_evidence/source_diff.json` | `5c44ee1a5f7c9179cb3f1ec3526ba6f113b55da81f83d5edb917e6c6889327bd` |
| `logs/a2_v10/probe_authorization_A1_inherited_20261127.json` | `49760a42facac7017b88db514b3beb09f8c7fb9b75458ac932a3a7bcc86d2d6c` |
| 原授权 `logs/a2_v10/probe_authorization_A_20261127.json` | `8afde6347fe88002da3a8bf542c080ff65f0ae52fbd47d8660cbd39fa67035e2` |

旧完整资源证据保持原字节，位于 `logs/a2_v10/formal_A_20261127_approved/resource_probe`：

- `selection.json`：`4f1e2c2738e1ad6cb8a2431f434c23634db4f232ac3023e56ad178d0dd7ecc04`。
- `n1024_dagger/result.json`：`37d422907bd3305ab10ac3da78223b49404b1b96e9446f6415055d8d27d0383e`。
- `n1024_ppo/result.json`：`6ccbbf3d5a0cb784cba91416721a6a9c6078536b3abd4674945d4447b2b072db`。

审核人重新读取全部295个身份域文件并独立重算：最终新指纹严格等于 `d90f857d…`；只将12个A1修改文件替回Git基线 `07fe14fe753ade340f3d64e06dbae51a9ad0bdb5` 的blob、排除11个A1新文件，其余272个文件维持当前实际字节，严格重建旧实际指纹：

`b48e3fd1601ff2926272ed014d846fd480fe24a8c56d77100d646f933b92991f`

因此不把Git基线误称为旧探测全部实际源码，也不把既有dirty/untracked误算为A1新增或擅自清理。Git差异清单共76项，其中包含已存在的旧工作区文件；其差异列表不能代替实际旧指纹重建。

## 不变项与唯一例外

42个核心/资产不变量中，41项实际字节一致。`env.py` 保留字节不同事实，唯一例外是明确批准的记录性 `frame.update`，不是“完全没改”。其AST证明只剔除指定 `V10Env.step` 内、`frame`建立后/`compute_reward`前的**一条精确匹配语句**，随后整模块AST必须与基线相等；额外字段、额外表达式、重复或换位置均不能被泛化忽略。

指定语句AST SHA为 `e63ff39081f9142c509b0908e70bc536276e73e44c7080997b2fda69163dab43`。它只clone记录yaw、足位置、top向上力及独立失败mask，不改变动作、物理、奖励或课程。新增临时存储仍须包含在新资源峰值中，不能据AST等价自行宣称资源已通过。

模型、观测45/5×45/3×40×30/3/35、六任务各32768条CPU FP32 replay、48步、3072逻辑批、微批≤256、PPO 5epoch/4batch、损失/奖励与机器人资产/控制物理保持原契约。验证器重新计算完整不变量集合和完整源码差异清单，删差异项或改证据摘要不能通过。

## 新资源协议与证书

- 原完整probe入口仍为1fill+5warm+20measure，未偷改为短测。
- A1明确命名 `A1-targeted-resource-v1`，每栈N1024、1fill+2warm+2measure，245760条；双栈491520条。完整replay、实际Adam更新、峰值不断档、磁盘SHA/CRC、释放旧存储、真实恢复与下一采样/RNG等价均执行公共完整core；probe学习状态丢弃。
- 接受的是**旧完整长测+规划显式批准的新短补证**。新两次measure不能重新证明20次测量稳定性，证书明确 `full_twenty_measurement_revalidation=false`，不改写或冒充旧完整证据。
- 父进程不驻GPU，两个child串行并共用GPU互斥锁；不启用512回退、增益搜索或自动重测。全局账本持久claim以protocol+批准contract SHA唯一绑定原序列/输出目录，child必须匹配原attempt及scope；即使极早失败费用少，换目录/序列也不能再买一次测量。
- 实际step按Q1预留/执行/uncertain计费；容量lease不是费用。父child共享scope，不双扣。失败只释放未用容量，executed/uncertain不退，成功短栈应charged=executed=245760、uncertain=0、updated=196608（填充49152不算梯度样本）。

首轮预审发现的缺失引用/重复stack风险已修复：验证器现在从真实dagger/ppo两个原始result、有效contract和全部引用重建canonical证书，并逐字段相等核验。必须恰好两个不同栈，selection行须等于对应原始文件；当前源码、SDS/裁定、N/物理、轮数、峰值门槛、全容量与完整阶段、恢复/采样摘要、实际费用及授权身份均需匹配。旧三文件、新两result、contract、proposal、源码差异、不变量、独立报告和裁定引用不可删除。unsigned selection内容摘要覆盖证书自引用两字段之外的全部语义，不能只改passed或N后复用证书。

## 授权承继与评估容量

生产承继授权经正常审批另存新文件，没有覆盖原授权。审核人只调用只读 `validate_authorization/global_report`：父原ID/SHA已registered，子ID不同，父文件SHA/ID、同ledger/SDS/scope/cap严格相等，子rulings SHA绑定当前A1裁定；没有旧rulings通配。新metadata首次注册要求prior charged与实际账本一致，重复承继幂等，活跃序列不能新旧并驻。

实际只读余额为：cap3833856、charged/executed3293184、uncertain0、updated3121152，剩余540672；没有pending或active sequence。A1双栈491520成功后理论余额49152，并非新增独立额度。本审核没有register/begin/reserve生产序列。继承前后历史费用不清零、不再次迁移。

阶段容量与step费分离：诊断251288；正式教师按每course尚需A/B全套预留；门禁/终验同样先核完整待执行工作。**最后一次预审发现**：旧 `_stage_capacity` 仅看resource-passed就减掉完成receipt，尚未核完整模型/病例身份，可能低估阶段required；根已接受并修复。

修后消费与减额共用 `_evaluation_identity_matches`：frozen文件SDS与实际manifest摘要、course/suite/reference/replicas、模型SHA（教师含A1）、code/rulings/physics及每组原始case唯一全集同时匹配。减额另要求durable result SHA和有效资源报告；partial teacher只在原准入路径可消费，不可抵扣完整未来容量。身份完整不代表行为合格，两类教师能力失败仍停止。合法恢复时完成candidate可仅预留尚余reference；坏身份/缺例/重复例不减额。

原全局评估cap1685390840、旧304100不退款；新零缓存全程最坏1685946228、潜在缺口555388，仍不足则停。此审核不宣称全程必足，也不将取消任务当cache。

## 验证与未验证

已读取实现方最终定向日志 `/tmp/v10_capacity_final_verified.log`：33 passed、1上游warning、36.24s，包含8类身份/病例负例且真实 `evaluate` 同时拒绝、合法恢复只余reference容量刚够、两类teacher失败停止。资源/承继主体此前定向 `/tmp/v10_A1_complete_final.log` 为54 passed、1warning、44.70s。审核人没有重复3GB测试或启动GPU。

根针对最终d90源码的统一全v10 CPU回归正在进行，最终结果由根补报；中间8daf的146 passed不计作d90通过。CPU源码/证据审核当前无阻断，实际进入物理前还须根确认最终统一回归通过、审批JSON/关联contract精确绑定本源与本报告，并走预算/资源门禁。

A1新GPU短测、112例诊断、正式教师A/B、DAgger/PPO及行为改善均未在本审核中验证。ETH原训练指令范围没有原cfg证据，normalizer非零std不能作为范围证明。

### 最终统一CPU回归补记

随后只读核对根最终日志 `logs/a2_v10/A1_engineering_20261127/final_v10_cpu.log`：**156 passed、5 skipped、1上游warning，84.65s**。对应最终冻结源码d90，而非中间8daf；显式关闭GPU/full-replay opt-in，未新增物理。本报告CPU工程批准的最后统一回归条件已满足，实际A1新资源短测/行为准入仍须按前述独立门禁执行。
