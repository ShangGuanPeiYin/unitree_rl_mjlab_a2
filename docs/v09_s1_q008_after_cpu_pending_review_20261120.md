# S1.1适配结束及Q008 after纯CPU部分审查（2026-11-20）

**当前不是整体工程通过，也不是S1.2晋级批准。** 真实813轮适配checkpoint的工程合同通过；已闭合after1500例的原件、轨迹、metrics、几何和费用核验完成，但新增CPU/GPU原容差例外仍待Q012裁定。生产`adapt_comparison.json`与adapt75视频尚未生成，不能提前签其工程flag或行为结论。before的Q011仅覆盖先前指定证据，未静默扩展到after。

本轮全部为CPU只读核验及新增本机证明；没有启动GPU、环境或物理，没有外发规划消息、CONT或暂停进程，没有修改原数据、原容差、生产源、旧签字报告/catalog、账本、共同日志或Git index，不提交。root因Q012待决持有原父进程，after子进程继续自然完成，其待决状态不在本报告中写成通过。

## 固定身份与适配checkpoint

固定run `a2-v09-s1-seed42-r3`、当前C9源码 `c9ee8789f6051bc6c36cbfc24f076deba6121d8ffedb48ce5d689e0a6b665ea4`、SDS `500ad32aaa946c927459427f3a65565b15d4b77fca4be214817121d15c742976`、唯一6400 `30ed23ba16f7e490afca001fbbecadb4a11ff4b11733a5ae7ac7d0c841f9a8b2`，沿用同Context清单 `0d158b266dde7daeefaf81bc5b270a4294a94f473203015ea8fd9d6df6380408`。

实际文件为 `checkpoints/s1-q010-formal-20261120-r1_S1.1_adapted_000019980288_00001637_0b1ff92df94045cbbd4f6bc788b48768.pt`，同字节SHA256 `3453350b921e81d0e3e568a3d6c93a6044b8c6b62edc6d3362083a0a0235d800`。元数据/C9/SDS/6400/同Context、RNG和角色租约、已写账本前缀及journal绑定均通过；所存前缀seq4912匹配完整stage/累计计费状态，不把当前合法追加尾部误判成篡改。

实际完成813×512×48=19980288学习控制样本，是2000万cap以内最大完整rollout数。Adam仍1组45参数，恰18项state各16260步=813×20；18项学习参数相对实际32c初始checkpoint变化，其余35项state逐项精确冻结，包含std和8个normalizer buffers。全部813条训练audit均与相应journal摘要匹配：每轮24576样本/20次Adam，loss/gradient及任务速度分组统计有限、分母完整。冻结s_V、LR3e-5及纯动作+0.1值蒸馏合同未变。RNG仅核保存格式和CPU局部roundtrip，没有GPU恢复。

独立证明 `checkpoint_0019980288_independent_proof_20261120.json` SHA256：`3401b0d5502db9674530d24f8e1609782931dd69cd308c33d6fe376da6abf74a`。这些是适配工程状态，不能据813轮、Adam步数或loss趋势宣称原能力恢复。

## after已闭合实际原件

`evidence/s1-q010-formal-20261120-r1_diag_after`全部75批/1500唯一病例已闭合，绑定适配checkpoint345335及冻结S1_DIAG manifest。独审按批读取75份PT、75几何证书，每批231个原编译地形对象及原contact索引重验；1500首次终态、完整raw metrics、实际命令/action历史、出生/终止位置和teacher/student非地图同状态输入核对通过。保存的15110个教师审计状态都做了CPU重算及FP64参考，输入/标签/输出有限，没有contact overflow/nonfinite工程掩码。

该项实际812240控制样本=首回合723322+padding88918，共3248960物理子步；75项操作为已知complete，case覆盖、真实timeout、padding与closed义务费用一致。审计状态每50控制步采样且排除已结束slot，不能把15110个审计状态数当成物理样本或训练分母。

全量证明 `q008_after_diagnostic_independent_numeric_pending_proof.json` SHA256：`a51fdd47c91a4663c3295ff8abd6d59f374f9d99352c2b0e4647a321ba770fee`，其中`accepted=false`保留未解除状态。

## 新数值例外，原式保持

逐分量仍用 `abs(cpu-saved) <= 1e-5 + 1e-4*abs(saved)`，没有由观察到的max反推容差，也没有重写保存GPU标签。

| 对照 | 超原容差 | 最大绝对差 |
| --- | ---: | ---: |
| teacher μ，全部15110保存状态 | 4991/181320分量 | 2.986192703e-4 |
| teacher V，全部15110保存状态 | 244/15110 | 1.049995423e-3 |
| 适配学生μ，每批固定初始观测 | 92/18000分量 | 6.377696991e-5 |
| 适配学生V，每批固定初始观测 | 0/1500 | 4.529953003e-5 |

首个实际批已提前报告：batch0/index100、完整20槽，teacher μ有5分量超原式；整批共162分量，未等整套结束才反馈。全量teacher共有703个异常快照，571个保存完整20槽、132个仅保存active子集；student固定初始观测的92个异常分量分布39批。**新增student范围也需处理，不能只核teacher后忽略它。** CPU对子集重算不等于复原原N20批推理；缺失inactive状态没有被历史重建。

CPU FP32相对FP64 teacher最大差为μ3.351957e-6、V4.984025e-6；保存GPU相对FP64最大差为μ2.987436e-4、V1.050603e-3。它们是参考差异，不证明具体算子根因，也不以“差小”代替验证。本报告没有做after GPU重放，没有为新范围签数值clearance。

异常位置清单 `q012_after_cpu_exception_inventory_20261120.json` SHA256：`28edf51167a9357c99a836e2d912c2604f9ab5cefbcf89118904829974386506`。按批/时刻/原slots记录full与active范围、原PT/字段SHA和异常统计，`approved=false`，仅为CPU证据清单，没有选择或授权GPU执行范围。

## 同数据四组CPU参考及未验证项

已独立计算初始32c、适配345335两个模型，各在同一before保存物理状态9366条和after15110条上前向，形成四组按75个任务/速度/停车组合的动作/值loss。保留原保存教师标签及s_V；两模型在相同数据集使用完全相同观测/标签/样本分母，全部loss有限，各数据集覆盖1500病例。另按冻结case顺序核对1500组before/after行为与相同stream，保留配对摘要。

参考文件 `q008_adaptation_four_way_cpu_reference_20261120.json` SHA256：`e343a52ab0e66ad8fee5b862474eeea814c75a4fdfc3379f81aa722ae1d352c7`。这仅是独立CPU参考，**尚未与未生成的production comparison核对，不等于接受其工程flag**；也不能把不同实际轨迹的loss或初始/适配行为趋势当作恢复。

仍缺：Q012新数值范围的正式裁定/核验闭合；实际`adapt_comparison.json`四组loss、配对与flags的核对；adapt75全部录像、精确终止帧、完整解码及费用闭合的审查。既有9d4是带运行时条件的批准，本部分报告不替代这些条件、不授权跳过暂停或进入S1.2，不宣称S1_RESTORED，不读取final成绩。后续实际结果应另建报告，保留本次pending证据和所有旧签字文件。日志及提交由root统一处理。
