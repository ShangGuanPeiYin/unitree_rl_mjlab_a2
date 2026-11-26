# A2 v1.0 正式工程验收汇总（2026-11-27）

**结论：修复后的工程代码与实际接线验收通过；资源、教师及策略行为总体验收未通过。** 不存在可交付的合格学生模型，本轮没有新增物理探测、教师评估或正式学习。

## 验收对象与可追溯身份

基线已推送提交25575edfb47168fa45c1367673fd85f0f64a3962，本轮在该版本上发现并修复两项工程缺口；不能将修后结果反称为基线本来正确。最终冻结源码SHA256为b48e3fd1601ff2926272ed014d846fd480fe24a8c56d77100d646f933b92991f，覆盖全部src Python及新版脚本/测试。SDS v1.1 SHA为eb95905ed24bf0674f94a1b02dab02256ee11e40a20ed1cd96eb24daeb1b49a0，原文未改。

逐项生产接线和证据见[v10_sds_evidence_map_20261127.md](v10_sds_evidence_map_20261127.md)，29项；独立审查者的[v10_final_independent_acceptance_20261127.md](v10_final_independent_acceptance_20261127.md)另列27项和135个已核实际行号链接。它重新阅读实际CLI→controller→worker→collector/env.step→reward/update→证书/导出路径，不仅复述纯函数测试或旧报告。

独立报告SHA256：6acb1311b30d33fb094dd24ba18cca8b13a7862d7ded5175cd521bd70569fd64。SDS、裁定、映射、独立报告、测试日志及身份记录的完整SHA列于同目录v10_final_acceptance_manifest_20261127.json。

## 实现与训练流程核对

正式入口为scripts/train_a2_v10.py。它串行组织工程身份→同N资源双栈→教师A/B→D1→D2→D3→唯一DAgger终验→P0→P1→唯一PPO终验→单学生导出。scripts/run_a2_v10.py的worker属于内部执行器，不应被当作具备全部行为门禁的独立正式入口。

| 核对项 | 实际设置及判定 |
|---|---|
| 两教师/学生 | 盲走8ee1ee1d…a4d6722、S2楼梯b3f31d1e…684831实文件SHA匹配；学生完整复制S2 Actor，教师冻结并在学生同一快照标注；部署仅学生 |
| 控制与输入 | 控制0.02s、物理4×0.005s；45本体/5×45历史/3×40×30地图/route3，12动作；q0+0.25a，现有raw clip=None；没有task或teacher特征输入Actor |
| 帧协议与物理 | 动作前command/phase/g锁存，动作后物理结算奖励；登顶动作事件计当前transition，下一动作换教师；重复observe不推进缓存；碰撞/地图同primitive，raw/reward/safe分离；H_route跌落面不跟随横向掉落 |
| 六任务/课程 | 比例20/10/15/15/10/30%；D1/D2/D3分别154/226/334组；D1宽平台整合，D2加入0.8m，D3完整路线；速度0.3/0.5/0.6m/s |
| 停车 | 100间隔线性减速或突停；HOLD1000间隔=20s；F_STOP重启5s，平台/楼梯停车重启5m；同帧失败优先 |
| DAgger | 六×32768 CPU FP32，3072逻辑批每任务512（当前/历史256/256），六任务等权动作MSE；48步rollout，1024每轮20Adam/512每轮10，微批≤256；更新后入库 |
| PPO | T_keep为合格DAgger学生；独立seed42 Critic/输出零，P0仅Critic；P1联合更新，std0.10初始化后投影[0.05,0.30]；5epoch×4批；gamma/lambda0.99/0.95，clip0.2，entropy0.003，keep1，LR3e-5固定 |
| 样本/保存 | D1=24576000、D2=49152000、D3=73728000、PPO=49152000（其中P0=4915200），学习合计196608000；1024最大轮数500/1000/1500/PPO1000，512加倍；每2457600保存，不借余额 |
| 门禁 | DAgger40/80/100%，PPO25/50/75/100%；同SHA A后B；每组20至少18、终验100至少95，并检查相对保持与窗口覆盖；最多两次PPO回退、不退款、不自动开新终验 |
| 恢复与证书 | 完整rollout/bootstrap/GAE/update后才暂停；恢复新物理回合；Q1 charged/executed/uncertain/updated独立、外账不退；Q3原checkpoint字节固定，先结果/哈希链事件/证书后snapshot、唯一ID幂等 |

PPO总奖励核对为r=0.02∑wᵢfᵢ+0.4Δx·I_cruise+2I_top+8I_success−10I_failure。连续项仅乘一次dt，位移不再乘dt；真实q/hard_limits计算软限位，真实运动、净空及四足safe_support计算stable，而非零指令或常数零。独立审核还对20连续项在巡航/停车两相位作实际代数核验。DAgger不以这些奖励更新Actor；PPO保持目标不是旧教师而是已合格统一学生。上述配方未被本次验收修改，未声称效果提升。

## 本次查出与修复的两项缺口

1. 原预算入口一次性要求全部最坏序列3833856，叠加737280历史使A配置无法启动。现按下一完整栈校验和独占容量：1024=1277952、512=638976；parent/child同scope共享，容量不等于charged，只有真正step扣费。不同scope互斥，pending不释放、uncertain不退。余量不足下一完整栈发布required/remaining/shortfall，不启动对应child。A虚拟CPU示例用完1024两栈后余540672，512缺98304正确停止；这不是实际新费用。
2. 原SDS8.1每任务长度/失败比例/标签覆盖没有持久出口。现真实collector→worker写per_task增量、完整终止分母、unfinished末状态gauge、每teacher expected/labeled/coverage、rollout_id及新物理session ID；不把48步当完整轨迹，不把未完重置算失败。PPO标签统计明确不适用（0/null），shape/finite坏标签在任何新step/预留前工程拒绝，不过滤缩分母。

本轮只修改runtime/resources/probe_budget及必要测试。损失、奖励、学习率、任务比例、物理、容量和保存周期未改；四个原有dirty文件本轮前后SHA完全相同，未纳入本次提交。

## 验证与复用范围

- 对25575ed逐字节比对：19个生产模块未改；对应既有q4_all_v10_final.log（83 passed、5 skip）及完整196608真实形状FP32/3.10GB CPU保存恢复8 pass证据有效复用。该83项不是本轮新组合重跑；5 skip不能称通过。
- 本轮仅重测缺口/身份：final_acceptance_gap_tests_20261127.log，24 passed、1上游Torch警告，28.65s。A/lease实施者另37 CPU pass，日志/保存联合14 pass、1完整CPU skip；数值来源和是否独立在报告中明确，不凑加重复测试数。
- 独立review_preflight重新核生产路径/配置、19模块和4 dirty/教师SHA，独立核20项实际双相位奖励代数；核原始新24项和旧3GB日志，不重跑昂贵GPU/物理或大payload。独立报告明确限定工程通过。
- 原裸pytest及全tests目录存在历史快照重名/legacy bootstrap收集冲突，仍未整体通过，不为变绿修改本任务外旧代码。

## 失败、未验证与执行阻断

| 层级 | 真实结论 |
|---|---|
| 本轮工程代码/实际接线 | 两项缺口修后通过，绑定本报告冻结源码；不是整体目标完成 |
| 原资源probe | 失败：1024最低host4283047936B、512最低2883108864B，均<4GiB；512即使25轮和恢复完整仍失败；N尚未选择 |
| 新资源probe | 未运行，本轮新增物理0；不能用CPU内存优化证据声称新资源达标 |
| 教师/正式训练/终验/播放 | 均未运行或未验证；正式学习0、教师suite0；无真实DAgger/PPO合格证书 |
| 历史费用 | charged=executed737280，updated663552，uncertain0，外部全局账本保留；迁移不产生新费用 |
| 执行授权 | 规划提出A累计cap3833856、余3096576，但创建生产预算文件被auto-review拒绝，因可信用户原始消息未明确该具体数字。规划确认数字来自推导/规划裁定，无用户数值原话；未创建文件、未绕过、未新探测 |

已具备的一般修改/push/正式训练授权持续有效。当前需要补充的是可信用户对probe累计3833856（包含已消耗737280）的明确数值确认；不是申请B4571136，也不是以工程通过替代资源/教师门禁。数值确认后才重新申请原预算落盘操作，再严格资源→教师→课程/终验；任何后续门禁失败仍按SDS停止并报告。
