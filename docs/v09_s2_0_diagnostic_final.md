# S2.0 诊断定稿报告（A2 v0.9，SDS v1.2 §4.3）

- 生成时间：2026-11-22
- 执行：DSH 执行主对话（A2继续训练）；实现组 A2-S2-IMPL；独立验收组 A2-S2-VERIFY
- 规范：`doc/sds/SDS_A2_v0.9_Sequential_Map_Walk_Stop_Robustness.md` v1.2，SHA256 `09fa2d4cdd858914c46a51fd52670c20cb20f2b6f3e76e6d266054574f64ccbe`
- S2 入口权重：`logs/a2_v09/s1_seed42_r3/checkpoints/s1_least.pt`，SHA256 `efa9caa8ce99e2cdff967212063e198b7ae68a6415727083b1db027f615ead24`（S1.2 主段 100,024,320 样本，用户批准进入 S2；不生成 `S1_RESTORED`）
- 不更新权重、无梯度更新；本报告只描述冻结入口权重的行为

> **DSH-Q037 §3 修订（2026-11-23）—— top 窗口径**
> 本报告 §3 的 `top_arrived` 是 §5.2 的 **0.4 m 四足入台几何代理**，不是 §4.2 的冻结安全登顶。按 Q037 要求，已从原始 trace 逐足精确重算冻结安全登顶（平台 geom 力 >5 N、0.042 m 内缩、逐足 2 帧、recent-50、`front+0.8`、`|y|≤0.20`、连续 5 帧、失败优先）：
> - 窄台 real（w0.8，15 例）：安全登顶 **15/15**，安全登顶→失败窗 **25–38 帧**，低头比例 **1.0**；
> - 宽台 real（w4.0，15 例）：**0/15**，且为几何证明（机身最大 x 8.1576 m < `front+0.8=8.8 m`）；
> - 旧 §3 的“top 窗 43–55 帧”保留为 **0.4 m 代理窗**，已被修订，不得再作为安全登顶窗引用；
> - 覆盖范围仅 30 个 `stairs_walk/real/N40` 病例；flat/platform/plane/H 系列不受影响。
> 派生件：`docs/v09_s2_0_top_recompute_q037.md`；代码/逐例/清单：`logs/a2_v09/s2_0/verification/q037_top_recompute_v2/`（script SHA `6946326e…`，per-case SHA `5141459e…`）。**v1**（`logs/a2_v09/s2/q036_verify/recompute_s2_0_top.py`）已被 v2 取代，其事件步经逐足精确重算证明不变。

## 1. 证据来源与覆盖

| 项 | 值 |
| --- | --- |
| 病例矩阵 | `S2_DIAG.json`（75 例）× real/plane 两分支 = **150 条**，manifest `fab7372e…` |
| 运行计划 | `S2_DIAG_run_plan.json`，150 条、30 批 |
| 完成度 | 150/150；0 个 runner 失败 |
| H 六例 | `h_series_case0.json`，manifest `22ea4111…`，6/6 completed |
| 汇总报告 | `logs/a2_v09/s2_0/reports/matrix-final-v7/report.json`（SHA `c741a033…`）与 `evidence_index.json`（`e12e8e1b…`） |
| 分组归因 | `group_attribution-v5.json`（`db49e1ee…`）、`group_attribution-v3.md` |
| 窄台边缘 | `platform_edge_analysis-v1.json`（`cc20c741…`） |
| Q031 输入 | `q031-platform-inputs-r1`（58 帧）、`q031-stairs-inputs-r1`（1,611 步） |
| 台账 | matrix 130,235 / 420,390；H 3,242 / 9,432 |

所有窗口统计剔除 `post_reset` 行、保留终止与失败事件；`pitch` 语义为 `pitch = -asin(2(wy-zx)) = asin((R_WB e_x)_z)`，**抬头为正、下俯（前倾）为负**（DSH-Q030/Q032）。

## 2. 150 条分组结果

| 任务 | 分支 | 平台宽 | 成功/失败 | 关键观测 |
| --- | --- | ---: | --- | --- |
| flat_walk | real / plane | — | **30/30 成功** | 20 s 走完，`route_delta` 约 5.3–11.3 m |
| stairs_walk | real | **0.8** | **0/15 成功** | 全部 `top_arrived`（15/15），`route_max` 9.1694–9.3117（顶后约 1 m），top 窗 43–55 帧、**15/15 全低头** |
| stairs_walk | real | 4.0 | 0/15 | **top 0/15**；nonfoot 13 / posture 1 / body 1；`route_max` 6.9499–8.1576 = "未确认登顶的登阶失败"（DSH-Q033） |
| stairs_walk | plane | 0.8 / 4.0 | 0/30 | **top 0/30**，全部未登顶；`route_max` 6.95–8.19 |
| platform_walk | real | **0.8** | **0/15** | 39–95 步失稳；末段两足越出 0.8 m 平台半宽、接触为 0 |
| platform_walk | real | 4.0 | **15/15 成功** | 走完 5 m，`route_max` 13.99–14.00 |
| platform_walk | plane | 0.8 / 4.0 | **30/30 成功** | 走完 5 m |

**核心复现**：40 级、宽 0.8、真实图，**15/15 全部爬完楼梯、`top_arrived` 成功、进入平台后约 1 m 前倾摔倒**，且 top 窗 15/15 全低头（`nose_down_fraction=1.0`）。这与用户现场"登顶、进入平台后才前倾摔倒"在阶段与姿态方向上一致。

## 3. 四窗（Q032 口径；top 窗部分已被 Q037 §3 修订，见文首横幅）

> **注（Q037 §3）**：以下 `top_arrived` 43–55 帧是 0.4 m 代理窗。冻结安全登顶口径下的修订数字为窄台 15/15、25–38 帧、低头 1.0，宽台 0/15；详见 `docs/v09_s2_0_top_recompute_q037.md`。本节原文保留作为旧代理记录。

`stairs_walk/real/w0.8`（15 例）：front_edge 58–70 帧（不要求全低头）、four_feet 44–56 帧（全低头）、top_arrived **43–55 帧（全低头）**、§6.5 巡航窗 **NA**（未达到四足后净前进 1 m 即 body failure）。
`platform_walk/real/w0.8`：front/four/top 均 38–94 帧（非全低头），§6.5 巡航窗 NA。

## 4. H 六例（定向回归，DSH-Q033：全部 `not_verified`）

| case | 状态 | 终止 | 说明 |
| --- | --- | --- | --- |
| H00 `84feecc2d400` | completed | physical_failure（posture） | 到达原失败位置附近（2.500 vs 2.279），但原失败类别为 `side`，**类别不匹配** |
| H01 `9ac0184ff02d` | completed | physical_failure（nonfoot） | 位置覆盖（2.717 vs 2.571），仅候选 |
| H03 `e826d1832915` | completed | **success** | 无失败，不构成复现 |
| H06 `ca6621521316` | completed | physical_failure（nonfoot） | 2.976 < 2.536？实际 reached 1.976 < 原 2.536，**未覆盖原失败位置** |
| H07 `4ad83f92f109` | completed | physical_failure（nonfoot，楼梯段） | 位置匹配，仅诊断匹配 |
| H10 `9e518aa2735b` | completed | physical_failure（nonfoot，楼梯段） | 位置匹配，仅诊断匹配 |

结论：六例均**未**达到"覆盖原失败位置且未复现"的正式复现标准，全部 `not_verified`；失败执行（早期 3 条 Q023 输入契约错误）已重跑并保留旧证据，不计为物理失败或完成复现。

## 5. Q031 同帧双输入反事实（证据等级 `pending`）

由于 mjlab 无法确定性重放/克隆状态（同进程 step1 即 qvel 分叉），按 DSH-Q031 采用"同一真实输入派生 plane 输入 + 双输入确定性推理"。

| 病例 | 触发/生效 | 阶段 | 样本 | 动作差 RMS | 动作差最大 |
| --- | --- | --- | ---: | ---: | ---: |
| platform real 0.8 | 出生帧生效 | activation | 1 | 0.814967 | 1.443449 |
| platform real 0.8 | — | post | 57 | 0.936107 | 3.576936 |
| stairs real 40 级 0.3 | trigger 1529 / activation 1530 | pre | 1530 | 0.0 | 0.0 |
| stairs real 40 级 0.3 | — | activation | 1 | 0.385288 | 0.580667 |
| stairs real 40 级 0.3 | — | post | 79 | 0.873917 | 3.599845 |

A/A 与 B/B 重复推理噪声均为 0；差异超过噪声底（`difference_significant=true`）。生效前输入逐元素完全相同、生效后仅有效格 height 变、valid/age 不变、替换值逐帧核验。

**限制**：这只能说明"该 real 状态下 Actor 对地图输入的即时响应"，物理轨迹配对仍 `not_proven`；不能据此推断"改图导致摔倒"或方向改变。

## 6. 窄平台边缘诊断

platform real 0.8（58 帧）：

- 台内有效格 37,078；台外有效格 32,522；**台外无效格 0**；台外 `height` 被 clip 到 −1 的占比 **97.68%**；
- 台内高度均值 −0.2749，台外 −0.9832；台内外 `age` 均值相同；
- 宽台（4.0）同例走完 5 m，窄台（0.8）失败时两足越出平台半宽、接触为 0。

**限制**：两个代表输入的 `invalid_cells` 均为 0，不能区分"台外低面有效回波"与"无回波无效格"；plane 整幅替换不能把某个边缘格认定为因果格。该结论作为 S2.1 显式诊断验证项，不设额外晋级阈值。

## 7. 宽 0.8 与 4.0 的差异（不可归因平台宽度）

两组楼梯局部碰撞盒完全相同（42 个），仅平台半宽 0.4 vs 2.0 不同；但两组使用**不同病例、不同 tile/origin 与不同 spawn x/y**（例 case0/0.3：−1.048649,−0.025433 vs −0.980255,−0.049676）。因此 `route_max` 6.95–8.16 与 9.17–9.31 的差异**不能归因平台宽度**；有效对照仅为同病例、同速度的 real/plane 分支配对。

## 8. 对 S2.1 的含义（不改配方）

1. 入口权重在 40 级窄台顶后会持续前倾失稳（15/15），这是 S2.1 的主要目标行为；
2. 平台起步在宽台可成功、窄台全失败，且 plane 全成功——窄台边缘真实高度信息是与失稳相关的输入因素之一，S2.1 的训练环境必须包含窄台边缘与宽台，验收必须分组；
3. 宽 4 m 的 40 级组在楼梯段即失败（top 0/15），属"未确认登顶的登阶失败"，不混入登顶后失稳结论；
4. 单例/单组不触发奖励或配方变化；按 SDS §6.2 原课程进入 S2.1。
