# 原 6400 S1.0 gate_A 单项证据独立验收

日期：2026-11-20。结论：**接受此次 gate_A 作为原 6400 的一套完整 S1.0 基线证据。** 此结论只验收已完成的真实数据和结账，不批准资源探测、S1.1/S1.2 学习或任何行为晋级，不改变原 S1.0 批准文件。

## 身份与固定分母

- run：`a2-v09-s1-seed42-r3`；数据目录 `logs/a2_v09/s1_seed42_r3/evidence/teacher_gate_A`。
- 生产源码 SHA：`44780cd5d6f9ee4029fb38b2f9348a872268e7bedf995d4b36e2e30ec2f2163f`；此次再次检查 322 个源文件全部匹配，batch 目录 source snapshot 与执行冻结 snapshot 相同。
- 原 6400 SHA：`30ed23ba16f7e490afca001fbbecadb4a11ff4b11733a5ae7ac7d0c841f9a8b2`。
- SDS SHA：`500ad32aaa946c927459427f3a65565b15d4b77fca4be214817121d15c742976`。
- gate_A manifest SHA：`94f85c887889a2444682085688d605017d5117e4b0becc03dd885b6fadd6fbe1`。
- 病例固定 75 组×20=1500。实际 75 批，每批 20 槽，全部病例唯一且与 manifest 集合、批次、顺序、slot、case_id、初始姿态和独立 streams 对应；没有按成功筛除病例。

## 全量独立核验内容

审核员使用 CPU、禁用 CUDA 对原文件逐批读取，未重新执行物理仿真，未修改任何原证据、源文件、账本或旧审核报告。

1. **75 个 PT 和 75 个独立 geometry JSON 全部检查。** 统一 PT 元组与 JSON 列表表示后证书内容一致，重算每个 certificate 摘要；原始 23 地块配置与冻结 v07 表相同。按证书记载的实际位置/尺寸重建只读描述对象，运行原 `index_geometry` 规则，231 个地形 box 全部通过，tile/step/top/support_bounds 逐项相等。75 批证书身份一致。此检查验证已保存的 compiled geometry 证据，不是本次重新构造仿真环境。
2. **1500 条首终态全部检查。** 每条原始轨迹 step 恰为 1 至 terminal_step 连续整数，其长度与 control_samples/end_control_step 一致且不超过对应 hard timeout；success/failure/timeout 三者恰有一个成立。初始位置、初始 yaw、终态带符号前进距离与原始张量对应。全部轨迹及初始观测张量有限，contact_overflow/nonfinite failure mask 均为 false。
3. **1500 例全部 metrics 从轨迹重算。** 使用冻结指标实现重算完整字典，与 episodes 中数据逐项相等；平地另按独立公式重算速度 RMSE 和接触滑移。完整窗口、quality failure、gait 周期、停车恢复和站立指标没有被成功分母过滤。
4. **75 批固定观测教师输出复核。** 初始 teacher heightmap 均为 3×48×32，history 为 5×45。CPU 重新载入唯一 6400，在保存的同一观测上重算 action/value，全部满足 `rtol=1e-4, atol=1e-5`；全体最大绝对差分别为 action `2.300739288330078e-05`、value `0.000164031982421875`。这是 CPU/GPU 浮点结果比较公差，不修改任何 SDS 行为阈值，也不宣称跨设备逐 bit 相等。
5. **独立重算汇总。** 全部 75 组成功数、终止原因、固定分母及 Wilson95 区间与报告相等；平地连续指标有效数、均值、步幅/步频中位数一致；楼梯带符号距离均值和 P10/P50/P90 一致。Markdown 的 75 组身份、成功数、Wilson 三位小数和原因计数逐行与 JSON 报告相同。
6. **费用独立交叉核对。** 75 个 reservation 全部 settled/complete，病例列表、20 槽、原 hard timeout、原教师 checkpoint、S1.0 归属正确。每批实际费用等于 `20×max(20例首终态步数)`，与 console 的对应批次及 ledger 一致；等待已结束槽的物理步没有被漏计。

评估 episodes 中 `teacher_control=False` 共 1500 条，是环境回合执行选择字段；本入口实际绕过该选择，明确直接调用 CLI 指定的 `policy=teacher`，向全部槽送入教师 action。审核依据是冻结评估执行路径、命令、summary、batch 身份及固定观测教师输出，不能把该环境字段误读成此次由学生执行。它也不能作为后续训练教师比例证据。

## 验收结果与结账

| 项目 | 独立重算结果 |
| --- | ---: |
| 固定病例总数 | 1500 |
| 成功 | 1051 |
| 物理/质量失败 | 448 |
| 超时 | 1 |
| 首终态轨迹控制样本 | 722,776 |
| 等待 padding 控制样本 | 94,624 |
| 实际计费控制样本 | 817,400 |
| 实际物理子步（×4） | 3,269,600 |
| 正式学习样本（本次账本采样） | 0 |

失败分类：quality_failure=13、side=102、nonfoot=256、posture=39、body=38，合计448；另 timeout=1。工程错误文件不存在，批次、原始轨迹、固定分母和结账未发现工程无效证据。

`teacher_gate_A` obligation 已 closed、reserved=0，1500 个 completed_cases 完整覆盖，spent=817,400，绑定原教师 SHA。原最坏预算 2,364,500 保留为 target/completed_upper；关闭后 limit 收敛为实际费用。独立验算账本从 header 至关闭事件 sequence=152 的全部 SHA 链，关闭事件 SHA：`7a0e364b61a5ccc7e55aa7e8afab3e48264dece86bfe3dc0fd1283f9b50a811d`。该已闭合前缀的原始文件字节 SHA：`5292b945e3b79071db17318eb6fe27df16283806fea3fcd9c77ac7bc7511d838`。B 套已追加的账本事件不改变此前缀。

## 可追溯文件身份

| 文件 | SHA256 |
| --- | --- |
| `evidence/teacher_gate_A/episodes.jsonl` | `366ba8cff25257d078f317a2e3bad398015c6abfde99335ea4c2096b1e30ceea` |
| `evidence/teacher_gate_A/summary.json` | `9273816d0c317f5a5630b50636a8317b7ce3d729b25d0598153aa0f8db2a5ce5` |
| `teacher_gate_A_report.json` | `13a4f54645933db67abc7d1a7e4a05d51c5d3158030166bb49c2583a429aa42f` |
| 新增 `teacher_gate_A_independent_verification_20261120.json` | `d588fed2fc1e60711580635ec9d313204214a85450c521ccd12e863a231303db` |

路径均相对于 run 目录；新增独审 JSON 保存全部逐批 PT/geometry 和主要输入/报告内容 SHA，以及本次计数和账本闭合前缀身份。新增文档及本机独审 JSON 的修改日志由主 agent 追加至 `logs/codex_changes.md`；本机运行证据不提交 Git。

## 不覆盖范围

此次接受的是原 6400 **gate_A 基线数据完整性及费用**，不是 v0.9 学生恢复、对比改善或 SDS 通过。未验收正在执行的 gate_B、三套 final 教师缓存、600 病例校准及 S1.0 finalize；未批准旧→新 Q010 兼容桥接、资源执行、资源通过、Q008 初始/后续真实证据、S1.1/S1.2 学习或 S2/S3。原资源入口阻断 R-RES-ENTRY-01 仍单列在资源审核报告中。

完整 S1.0 仍需剩余原始证据和义务结账后逐项验收；不得用本单项报告代替 baseline_complete 或修改任何接受门槛。
