# 原 6400 S1.0 gate_B 单项证据独立验收

日期：2026-11-20。结论：**接受此次 gate_B 作为原 6400 的一套完整 S1.0 基线证据。** 先只读确认 summary 已生成、`teacher_gate_B` obligation 已关闭，才开始此轮核验。此次仅新增 B 独审 JSON 和本报告，不改 A、旧 S1.0 签字、生产源码或账本，不运行物理仿真，不读取或展示任何 final 教师行为结果。

## 身份和范围

- run：`a2-v09-s1-seed42-r3`；B 证据目录 `logs/a2_v09/s1_seed42_r3/evidence/teacher_gate_B`。
- 生产源码 SHA：`44780cd5d6f9ee4029fb38b2f9348a872268e7bedf995d4b36e2e30ec2f2163f`；322 个源文件再次逐项匹配，B source snapshot 与执行冻结 snapshot 一致。
- 唯一原教师 6400 SHA：`30ed23ba16f7e490afca001fbbecadb4a11ff4b11733a5ae7ac7d0c841f9a8b2`。
- SDS SHA：`500ad32aaa946c927459427f3a65565b15d4b77fca4be214817121d15c742976`。
- B manifest SHA：`2024a49022bc169f33ccc121654f50c8c2ea116b791c74dfd5f3bcb9f86ac510`。

使用 CPU、`CUDA_VISIBLE_DEVICES=''`，逐批读取 B 的全部 75 PT、75 geometry 和 1500 首终态，独立重算，再与主 agent 新生成的 B JSON/Markdown 报告对照。审查过程中 ledger 的读取只用于已闭合 B 前缀与 B 义务/操作结算，不读取 final PT、episodes、summary 或派生行为报告。

## 全量核验结果

1. **固定分母及首终态：** 75 组各20例，1500 case SHA 唯一且完整等于 manifest，批次/顺序/slot/case_id、初始位置与 yaw、streams、原教师 checkpoint 和 S1.0 归属一致。每条轨迹从 step1 连续至 terminal_step，长度与 control_samples/end_control_step 一致且不越 hard timeout；success/failure/timeout 恰有一项成立。终态带符号前进距离与保存的位置张量对应。
2. **实际几何证据：** 75 对 PT 内证书和独立 JSON 统一元组/列表表示后内容一致，certificate SHA 全部重算通过。冻结23地块原表一致；按证书记载的 compiled 位置/尺寸构造只读描述对象，原 `index_geometry` 验证每批231个地形box，通过 tile/step/top/support_bounds 全项比较。75份证书具有同一内容身份。本次未重建物理环境，验证的是生产运行保存的几何证据及其规则一致性。
3. **全部指标：** 1500例完整 metrics 字典从原始轨迹通过冻结实现重新计算，逐项相等；平地速度 RMSE 和接触滑移另用独立公式交叉核算。轨迹和固定初始观测张量全部有限，所有病例的 nonfinite/contact_overflow failure mask 为false，没有工程错误文件。
4. **教师输出：** 75批固定初始观测均为原教师3×48×32地图、5×45历史；CPU重新载入唯一6400输出 action/value 与保存值全部满足 `rtol=1e-4, atol=1e-5`。最大绝对差分别为 action `2.288818359375e-05`、value `0.00018644332885742188`。这是跨CPU/GPU浮点核算公差，不改变任何行为阈值，也不声称逐bit相等。
5. **独立汇总与主报告对照：** 先从原始数据按75组自行形成成功/失败/超时、原因、Wilson95、连续指标有效数和均值、步幅/步频中位数、楼梯距离均值和P10/P50/P90，再与 `teacher_gate_B_report.json` 全组对照相等；Markdown75行组身份、成功数、Wilson三位小数和原因计数相等。分母不随成功数改变，缺测不填零。
6. **独立费用核算：** 75个reserve全部settled/complete，每批20槽、原最大timeout及实际步骤一致。实际批费用等于 `20×max(该批20例首终态步数)`，与 console 和 ledger 相等，包括提前结束槽等待期间的物理padding。1500个completed_cases覆盖完整，obligation closed、reserved=0、绑定原6400。

原始 episodes 中的 `teacher_control` 属环境回合执行选择字段；该评估入口直接执行指定teacher策略，不使用训练回合的选择逻辑。不能以这个环境字段判断此次是学生执行，也不能拿它验证后续训练的教师回合比例。此处教师身份依据冻结CLI执行路径、命令、summary、batch身份与同观测教师输出。

## 实际结果和账本闭合

| 项目 | 独立结果 |
| --- | ---: |
| 完整病例 | 1500 |
| 成功 | 1037 |
| 物理/质量失败 | 460 |
| 超时 | 3 |
| 首终态轨迹控制样本 | 726,201 |
| 等待padding控制样本 | 143,599 |
| 实际计费控制样本 | 869,800 |
| 实际物理子步（×4） | 3,479,200 |
| 正式学习样本（本次账本采样） | 0 |

失败细分：quality_failure=16、side=90、posture=57、nonfoot=253、body=44，合计460；另timeout=3。未发现使此次B套无效的工程异常。该结果描述原6400在固定B集上的表现，不表示v0.9学生改进或通过。

独立验算 ledger header 至 B 关闭事件 sequence303 的全部SHA链。关闭事件SHA为 `bdbb46a4fe18ac5354b66f76c42dc55e3fda3588014784d5365313c3b658086a`；已闭合前缀原始字节SHA为 `0904eab8b305c74e44b3d90ef13b2b5ee6c4897484943dd87f8708d1804275f0`。后来追加的其他操作不改变这个已闭合B前缀。

## 证据内容身份

以下路径相对于 `logs/a2_v09/s1_seed42_r3`：

| 文件 | SHA256 |
| --- | --- |
| `evidence/teacher_gate_B/episodes.jsonl` | `a08df17facad76cb6d6292213c24002962835188e8d2caf753481dc37d7aadf1` |
| `evidence/teacher_gate_B/summary.json` | `94f4b7c572b1ab144a48ae90f0d38b1d5293fc5a69214c8cee697415531f01ff` |
| `teacher_gate_B_report.json` | `a5d432afc317fbaa6df0989e8bb02e4a1085490c8cc717c3d7fcc1d03ec9410c` |
| 新增 `teacher_gate_B_independent_verification_20261120.json` | `12ce3cbe642a21f0c977c92bf14c2e15eea4fc10b5c9d18fb9909fb44765a923` |

新增独审JSON包含75批PT/geometry逐文件SHA、原始输入和报告SHA、独立75组汇总、首终态/padding及闭合前缀身份。修改日志由主agent追加至 `logs/codex_changes.md`，本机证据JSON不提交Git。

## 不覆盖范围

仅接受 **B套原教师基线数据的完整性、汇总和计费**。不扩展到final教师数据、600病例校准、S1.0 finalize、Q010旧新兼容、资源执行/通过、Q008真实初始或后续证据、S1.1/S1.2学习或S2/S3。final教师结果继续封存。本报告不替代 baseline_complete、独立资源批准或学习批准，不更改任何课程、奖励、损失或验收门槛。
