# V9 正式独立复核（阶段版）：S2.0 全量矩阵、H 六例、窄台边缘与 S2.1 准入

- 复核角色：A2-S2-VERIFY（独立于实现方）
- 复核裁定：**DSH-Q033**（commit `c122666`）；同时引用 Q030/Q031/Q032
- 复核对象：`matrix-final-v4/report.json`、150 条 S2 evidence、6 条 H evidence、`q031-platform-inputs-r1`/`q031-stairs-inputs-r1`、probe/admission、ledger
- 方式：**纯 CPU、只读**；未运行 GPU、未推进物理、未训练、未修改实现/测试/SDS/证据/ledger/checkpoint，未做 git 写入
- 状态：**阶段版 + post-rerun 快照**。H 三条原 `runner_failed` 已在 21:10–21:16 重跑（H01 910 步 physical nonfoot、H02 304 步 success、H03 274 步 nonfoot）；`matrix-final-v4` 尚未把这些重跑并入报告。1024 probe 已启动但**在构造 S2TrainEnv 时失败**（`_sample_tiles()` 参数错误），无 probe_report/admission。Q033-3 仍无边缘专用 artifact。
- 产物：本报告 + `logs/a2_v09/s2_0/verification/full_matrix_audit_20261122_r9.json`

## 0. 结论摘要

| 项目 | 结论 |
|---|---|
| S2 分组计数 | 除一处 top 窗范围外，与父会话一致；flat 30/30 成功、platform 组匹配、stairs 60/60 失败 |
| `stairs_walk` real 宽 0.8 | 15/15 失败、15/15 top、route 9.1694–9.3117；top 窗实际 **43–55 帧**，不是 46–55；15/15 top 窗 nose_down=1.0 |
| `stairs_walk` real 宽 4.0 | 15/15 失败、**top 0/15**、nonfoot 13 / posture 1 / body 1、route 6.9500–8.1576；4/15 route>=8 但仍 top=false；分类为“未确认登顶的登阶失败”正确 |
| `stairs_walk` plane | 30/30 失败、**top 0/30**；0.8 组 15/15 越过 front x=8 但未 top，4.0 组 3/15 越过 front；不是“多数未到顶”而是**全部未 top** |
| platform / flat | real 0.8 15/15 失败、real 4.0 0/15 失败、plane 30/30 成功、flat 30/30 成功：**通过** |
| 几何差异归因 | 楼梯局部几何相同（42 boxes，tread/ground 相同，只有 platform half width 0.4 vs 2.0）；tile 1/2、origin y=12/28、spawn x/y 均不同；**不能把 7.0 vs 9.2 归因于平台宽度**，是 tile/birth/病例未配对的混杂 |
| H 六例 | 原 3 runner_failed 确为 Q023 输入契约错误（349/284/284），不是物理失败；3 completed 为 physical_failure（H00 posture、H04/H05 nonfoot）。**v4 的 5 covered/1 not_covered 与 Q033“六例均 not_verified”冲突**；21:10 重跑结果：H01 910 步 nonfoot physical、H02 304 步 success、H03 274 步 nonfoot（H03 1.976<2.536，未覆盖） |
| H 重跑/预算 | 3 runner_failed 已重跑：H01 physical、H02 success、H03 nonfoot（未覆盖）；H 已 commit 3,242，余 6,190；若还要再跑 H02/H03，worst-case 2,621 仍在余量内；六例全部重跑仍须另算 |
| 窄台边缘 | 有效高度替换/valid/age/动作敏感度可复算；但面板初步的 0.797 m 与 -0.3038 不是同一帧，独立最大差 0.7991；两个代表输入均无 invalid 格，无法验证“低面回波 vs 无效格分列” |
| 1024 probe/admission | **probe 启动即失败**：`S2TrainEnv.__init__._sample_tiles() takes 1 positional argument but 2 were given`；无 probe_report/admission；3,709,184 预留未释放，训练不得启动 |
| 预算 | matrix 130,235 不变；H 实际 **3,242**（原 1,754 + 重跑 1,488）；debug **11,905**；总实际 **145,382**，比用户 142,224 多 3,158（H 重跑 1,488 + q031 platform 59 + q031 stairs 1,611）；probe 另预留 3,709,184 未释放；无矩阵重复 |

## 1. S2 全量独立统计

数据源：`matrix-final-v4/report.json` 的 evidence_index + 逐例 `smoke_summary.json` + `trace.jsonl`；用 trace 的 `post_reset` 排除出生行、保留 terminal 事件，独立重算 route/window。

| 组 | n | 失败 | top | route_max 范围 | 失败类型 |
|---|---:|---:|---:|---|---|
| flat real w8.0 | 15 | 0 | 0 | 5.3761–11.3078 | — |
| flat plane w8.0 | 15 | 0 | 0 | 5.2690–11.2272 | — |
| stairs real w0.8 | 15 | 15 | 15 | 9.1694–9.3117 | body 12 / nonfoot 3 |
| stairs real w4.0 | 15 | 15 | 0 | 6.9500–8.1576 | nonfoot 13 / posture 1 / body 1 |
| stairs plane w0.8 | 15 | 15 | 0 | 8.1303–8.1936 | nonfoot 15 |
| stairs plane w4.0 | 15 | 15 | 0 | 6.9508–8.1608 | nonfoot 15 |
| platform real w0.8 | 15 | 15 | 15 | 9.0007–9.1595 | posture 11 / nonfoot 4 |
| platform real w4.0 | 15 | 0 | 15 | 13.9898–13.9998 | — |
| platform plane w0.8 | 15 | 0 | 15 | 13.9882–13.9996 | — |
| platform plane w4.0 | 15 | 0 | 15 | 13.9909–13.9999 | — |

- S2 `failure=true` 总数为 **75**，`failure=false` **75**；与 `matrix-final-v4.failure_counts` 的 physical_failure 75/completed 75 一致。
- `matrix-final-v4` 的 `metrics_error=0`、`h_cases` 有 6 条，已修复 `final-r2` 的 H 缺失；但 H 覆盖结论仍有 Q033 冲突（见 §4）。

### 1.1 与父会话数字不符的一处

父会话给 `stairs_walk real 宽 0.8 top_arrived 窗 46–55 帧`；独立复算和 `matrix-final-v4` 都是 **43–55 帧**：

| case / speed | top 窗帧 |
|---|---:|
| 0 / 0.3, 0.5, 0.6 | 48, 49, 54 |
| 1 / 0.3, 0.5, 0.6 | 50, 51, 54 |
| 2 / 0.3, 0.5, 0.6 | 46, 54, 55 |
| 3 / 0.3, 0.5, 0.6 | 44, 45, 43 |
| 4 / 0.3, 0.5, 0.6 | 49, 44, 43 |

因此 46–55 不是可复现的全组范围；43/44/45 帧的 case 存在。all 15 的 `nose_down_fraction=1.0`，**“登顶后持续前倾”在全部 15 例成立**。

## 2. 姿态与失败位置

### 2.1 real 宽 0.8

15/15 `top_arrived=true`；top 窗均值约 −20.7°～−25.8°，`nose_down_fraction=1.0`；最后有效 pitch −59.79°～−37.38°。这是**登顶后持续低头/前倾**，不是单例。

### 2.2 real 宽 4.0 与 plane

| 组 | top | route_max | 越过 front x=8 | 失败时姿态/位置 |
|---|---:|---|---:|---|
| real w4.0 | 0/15 | 6.9500–8.1576 | 4/15 | 11/15 在 front 前失败；4/15 到/过 front 但 unconfirmed top；last pitch +11.6°～+68.6° |
| plane w0.8 | 0/15 | 8.1303–8.1936 | 15/15 | 全部到 front 之后但在台缘/未锁存 top 处 nonfoot 失败；last pitch +12.8°～+20.3° |
| plane w4.0 | 0/15 | 6.9508–8.1608 | 3/15 | 12/15 front 前失败；3/15 到/过 front 但 top=false；last pitch +15.5°～+44.5° |

- **没有一例是“登顶后失败”**：宽 4.0/plane 组的 `top_arrived` 全为 false；宽 4.0 的 4 个 route>=8 例也不能记 top（Q033-2）。
- 宽 4.0 的“未确认登顶的登阶失败”分层正确：nonfoot 13、posture/body 各 1，route 6.95–8.16；没有把 route 接近/超过 8 m 误当 top。

## 3. 0.8 vs 4.0 楼梯差异：几何、tile、出生

CPU 复算 `s2_terrain.terrain_specs()`：

| 项 | 宽 0.8 | 宽 4.0 |
|---|---|---|
| N | 40 | 40 |
| 局部 collision boxes | **42** | **42** |
| ground + 40 treads | 完全相同（局部坐标） | 完全相同（局部坐标） |
| platform half width | 0.4 | 2.0 |
| tile_id | 1 | 2 |
| frame_origin y | 12.0 | 28.0 |
| 例：case0/speed0.3 spawn x,y | −1.048649, −0.025433 | −0.980255, −0.049676 |
| trace map_valid fraction | 1.0 | 1.0 |

结论：

1. **不是地形/楼梯几何生成差异**：两种宽度都是 40 级、h=0.2、d=0.2、stair width=0.8；局部 tread/ground 完全相同，只有 top platform 半宽不同。
2. **确实是 tile/出生差异**：tile 1 vs 2、origin y 12 vs 28，且同 case_id/speed 的 spawn x/y 不同；这些病例不是配对样本。
3. **平台宽度不是已证因果**：宽 4.0 的 11/15 在 x<8 的楼梯段失败，失败位置早于平台；平台宽度最多可能通过前方 heightmap 影响策略，但当前没有控制变量证据。0.8 vs 4.0 的 7.0/9.2 差异是 **tile+spawn+platform width 的混杂**，不能单归因平台宽度。
4. 窄台边缘诊断必须用同 spawn/tile 的配对或同帧输入，不能把这 15 vs 15 当几何因果。

## 4. H 六例（Q033）与重跑结果

### 4.1 原 runner 退出原因（matrix-final-v4）

| lesson | 原 matrix status | steps | smoke error | trace physical |
|---|---|---:|---|---|
| H00 | completed | 446 | — | physical_failure, posture |
| H01 | failed | 349 | `ValueError('real map failed Q023 frame proof at Actor step 349')` | failure=false/terminal=false/running |
| H02 | failed | 284 | 同上，Actor step 284 | failure=false/terminal=false/running |
| H03 | failed | 284 | 同上，Actor step 284 | failure=false/terminal=false/running |
| H04 | completed | 193 | — | physical_failure, nonfoot |
| H05 | completed | 198 | — | physical_failure, nonfoot |

- H01/H02/H03 的 `runner.log` 明确 `status=failed`、`error=ValueError(...Q023 frame proof...)`；`map_evidence.frame_check_failures=1`；trace 没有 failure/terminal 事件。**是输入契约校验（Q023 frame proof）错误，不是物理失败，也不是完成复现**。
- H00/H04/H05 的 `termination=physical_failure`：H00 posture、H04/H05 nonfoot；failure event 与 summary 一致。
- `matrix-final-v4.failure_counts` 是 **S2-only**（runner_failed=0）；H 的 3 个 runner_failed 只在 `h_cases` 中，不得读成整矩阵“无 runner 失败”。

### 4.2 v4 failure_coverage 与 Q033

`matrix-final-v4` 当前计算（重跑前）：

| lesson | route covered | surface_match | v4 status | Q033 required |
|---|---|---|---:|---|
| H00 | true (2.500 >= 2.2785) | true | covered | **not_verified** |
| H01 | true (2.539 vs 2.571，差 0.032) | false (Q023) | covered | **not_verified** |
| H02 | false (2.597 vs 2.657，差 0.0595) | false | not_covered | **not_verified** |
| H03 | true (2.601 vs 2.536) | false (Q023) | covered | **not_verified** |
| H04 | true (1.020 vs 1.033，差 0.013) | true | covered | **not_verified** |
| H05 | true (1.20305 vs 1.20368，差 0.00063) | true | covered | **not_verified** |

- v4 的 5 covered / 1 not_covered / 0 not_verified **违反 Q033-6“六例 failure_coverage 当前均 not_verified”**。H04/H05 即使位置/类别看起来匹配，也只能作为诊断值，不能写成 verified。
- H00 的 v4 只有“位置覆盖”这一项真正成立（表面虽同平台，但原始失败是 `side`，当前是 `posture`，类别不一致），不能算失败复现；若按四个判定数，只有 `covered` 一项，其余不成立。
- H01/H03 虽有空间覆盖，但 Q023 runner 失败，物理结果未产生；H02 连空间覆盖也未达到。原六例均为 not_verified。

### 4.3 21:10 后的 H 重跑（新增证据）

三条原 runner_failed 已重跑，证据在 `logs/a2_v09/s2_0/evidence/h-q033-r1-*`：

| lesson | 重跑 steps | termination | route_max | 原 route | 判断 |
|---|---:|---|---:|---:|---|
| H01 | 910 | physical_failure / nonfoot | 2.7171 | 2.5711 | 同 nonfoot 类别，位置覆盖；可作为 candidate，但尚无正式更新 |
| H02 | 304 | **success / legacy_v07_success** | 2.6313 | 2.6570 | 不是物理失败，原 posture/side 未复现；not_covered |
| H03 | 274 | physical_failure / nonfoot | 1.9762 | 2.5361 | 位置差 0.56 m、top=false，类别不同；not_covered |

- H01 是三条重跑中唯一支持“原失败类别+位置”的候选；H02 直接成功，H03 在更早位置以 nonfoot 结束，均不能算复现。
- H00/H04/H05 未重跑；H00 类别仍 side vs posture；H04/H05 类别/位置匹配但 Q033 原 coverage 仍 not_verified。
- **正式 H 状态**：原六例 coverage 全部 not_verified；重跑后 H01 为 candidate，H02/H03/H00/H04/H05 仍未验证；报告 v4 尚未纳入这些重跑。
- H 预算：原 commit 1,754 + 重跑 1,488 = **3,242**，cap 9,432，余 **6,190**；H reserve 累计 11,797（含已释放旧 reserve），实际 committed 3,242；无重复。
- 若还要再跑 H02/H03，worst-case 1306+1315=2,621，仍在 6,190 内；若六例全部按 worst-case，需另算且可能超余量。

## 5. 窄台边缘诊断（Q033-3）

### 5.1 输入与同帧证明

`q031-platform-inputs-r1`：58 帧、base z 8.3014–8.4654；`q031-stairs-inputs-r1`：1610 帧、base≈0.4。两者均：

- `same_frame_runner=true`；
- `before_exact_equal=true`（stairs 1530 帧 pre-trigger）；
- `after_valid_age_equal=true`、`after_only_valid_height_changed=true`、`after_replacement_value_checked=true`；
- 独立用 `apply_plane_height_channel`/`verify_policy_frame` 对 58/1610 帧重放，全部 `passed`。

### 5.2 格点投影与高度

高度语义为 `clip((z_surface - base_sample_z)/1.5, -1, 1)`；grid 为 body 系 x∈[-0.775,1.175]、y∈[-0.725,0.725]，step 0.05。

平台例：

- platform top=8.0；frame 0 base 8.400000 → 期望 plane 高 (8−8.4)/1.5 = −0.266667；
- frame 20 base 8.455650 → 期望 **−0.3037669**，与输入中 plane 高度一致；
- 外侧 ground/低面：real height=−1.0（clipped），并非 invalid；平台宽 0.8，y 外缘格（y index 0–6、23–29）共 560 格，valid=1；real low 格每帧约 558–560（外缘 low 497–520，其余 low 在台内边缘/落脚附近）。
- valid/age 在所有帧 real/plane 完全相同；动作敏感度：
  - activation RMS **0.8149673**、max **1.4434488**；
  - post RMS **0.9361070**、max **3.5769358**；
  - 与父会话“0.815/1.44、0.936/3.58”一致。
- 首帧/失败附近 `foot_on_platform` 与台面几何一致（四足/至少支撑足在 0.8 m 台面内；q031 首帧足接触 2/4，平台支撑全为 4）。

### 5.3 与父会话初步量化的差异

父会话给出：real 外缘 `height=-1.0` vs plane `-0.3038`，差最大 **0.797**，valid/age 相同。

独立复算：

- plane −0.3038 出现在 frame 20（base 8.45565），该帧 real −1 的差为 **0.6962**；
- 全平台最大 |real−plane| = **0.7991**，在 frame 34（plane −0.2009，base 8.30136）；
- 0.79737 出现在 frame 35（plane −0.20263），不是 −0.3038 同帧。

因此“−0.3038 与 0.797 同一外缘/同一帧”的说法不成立；**0.797 是父会话的约数/近似，且不能与 −0.3038 绑定**。valid/age 相同和动作敏感度成立。

### 5.4 invalid 格与限制

- 两个代表输入的所有帧均 **invalid_cells=0**（platform/stairs 均全 valid），因此“台外真实低面回波”与“无效格”在这两个例中**没有同时出现**；real low 格是 valid 的 −1，而不是 invalid；无法用当前样本验证二者分列。
- 当前 Q031 报告 `evidence_grade=pending`、`physical_trajectory=not_proven`；statement 明确只是同帧输入替换的即时动作响应，未做物理轨迹配对。
- **plane 整幅替换不能单独识别边缘格因果**：这一点被报告和 evidence grade 守住；不得把 0.815/1.44 或 0.936/3.58 解释为“边缘格导致摔倒”的因果。
- 当前**没有独立的 Q033-3 边缘专用 artifact**；现有 `counterfactual_report` 是整幅 plane 替换，不是“台内/台外/无回波”分列报告。结论只能标 pending。

## 6. S2.1 准入（Q033-8）

### 6.1 1024 完整 probe 失败

代码要求（`s2_resources.REQUIREMENTS`/`run_probe`）：

```text
peak <= 7168 MiB; host free >= 4096 MiB
warmups=5; measurements=50; restore_checks=1
evaluation_envs=100; evaluation_steps=9472
video_envs=1; video_steps=200
rollout=48; independent_copy=true
```

- `probe_plan(1024)` control samples = **3,709,184**；ledger 已 reserve 3,709,184（`resource_probe_1024`），无 commit。
- 21:13–21:14 的 probe 启动失败，`logs/a2_v09/s2/probe/1024-r1.log` traceback：

```text
TypeError: S2TrainEnv.__init__.<locals>._sample_tiles() takes 1 positional argument but 2 were given
```

- 失败发生在 `S2TrainEnv(...)` 构造/reset，尚未进入 warmup/measurement/evaluation/video/restore；没有 `probe_report.json`、没有 admission。
- `independent_copy` 仅记录调用参数，仍需 GPU 窗口确认独立副本语义。
- 当前不能启动训练；必须先修 `_sample_tiles` 绑定签名并重新完成完整 1024 probe。

### 6.2 512 fallback

- `choose_resource` 明确要求先有 1024 探测，且 1024 `admitted=false` 才能选 512；当前没有 1024 失败报告（只有启动崩溃），**不能以 512 或降并行替代完整 1024 探测**。
- admission 要求：`a2_v09_s2_1_admission_v1`、`admitted=true`、`num_envs=1024`、probe SHA 一致、independent review SHA 64 hex；当前均缺失。
- probe 的 3,709,184 reserve 目前仍未 release；这是预算上的未结项，不是已花步数。

### 6.3 与 H 的关系

- H 重跑不阻塞训练（Q033），但当前训练阻塞在 **probe 构造失败**；H 结论仍按 §4 如实标注，H01 只是 candidate。

## 7. 预算核对（21:16 快照）

| 项 | 用户口径 | 独立账本 | 说明 |
|---|---:|---:|---|
| matrix net | 130,235 | **130,235** | full 127,058 + limit2 3,177 |
| S2 completed steps | — | **127,038** | full ledger 多 20 = 20 条 r1 platform 首步失败 |
| H original | 1,754 | **1,754** | 446+349+284+284+193+198 |
| H rerun | — | **1,488** | H01 910 + H02 304 + H03 274 |
| H committed total | 1,754 | **3,242** | cap 9,432，余 **6,190** |
| debug | 10,235 | **11,905** | q031 platform 59 + q031 stairs 1,611 + 原 10,235 |
| total spent | 142,224 | **145,382** | 多 3,158 = H 重跑 1,488 + q031 1,670 |
| probe reserve | — | **3,709,184 outstanding** | probe 构造失败，未 commit/release |
| matrix limit | 420,390 | 420,390 | 使用率约 31% |

- matrix full 127,058 = S2 完成 127,038 + 20 个 platform r1 首步失败；limit2 3,177 另计；无 full 重复/漏记。
- H 3,242 含 1,488 重跑；原 H 失败三条未按物理失败计；H 预算仍足以下一轮 H02/H03（2,621）。
- debug 11,905 的增量：q031-platform 59、q031-stairs 1,611；用户 10,235 是此前快照，不是重复计费。
- probe 的 3,709,184 仍为 outstanding reserve；未产生 control samples，但预算上不能当已释放。

## 8. 发现清单

| ID | 级别 | 内容 |
|---|---|---|
| P-V9-1 | 通过 | S2 分组计数/route 与父会话一致（除 top 窗范围） |
| B-V9-1 | 不符 | real w0.8 top 窗实际 43–55，不是 46–55；report v4 也是 43–55 |
| P-V9-2 | 通过 | 宽 4 m real top=0/15、nonfoot13/posture1/body1、route6.95–8.16；分类正确，未把 route>=8 当 top |
| P-V9-3 | 通过 | real w0.8 全部 15 例 top 窗 nose_down=1.0；宽4/plane 无 top，非登顶后 |
| B-V9-2 | 方法学/阻断 | 0.8 vs 4.0 局部楼梯几何相同；tile/origin/spawn 不同，病例未配对，7.0 vs9.2 不能归因平台宽度 |
| B-V9-3 | 阻断（Q033） | v4 H failure_coverage 5 covered/1 not_covered，违反 Q033 六例均 not_verified；H00 类别也不匹配 |
| P-V9-4 | 通过 | 原 H 3 runner_failed 原因 Q023 输入契约；3 completed physical_failure；progress 一致 |
| P-V9-5 | 阶段 | H01 重跑 physical nonfoot 候选覆盖；H02 success、H03 nonfoot 位置不足；不能把三条都算复现 |
| B-V9-5 | 阻断/准入 | 1024 probe 构造失败 `_sample_tiles`；无 probe_report/admission；3,709,184 reserve 未释放；不得训练/不得 512 替代 |
| P-V9-6 | 通过/受限 | 边缘输入 valid/age/height/动作敏感度可复算；全幅替换限制守住 |
| B-V9-4 | 待补 | 初步“−0.3038 vs 0.797”不同帧；独立 max 0.7991；两个代表输入无 invalid 格，无法验证低面回波 vs 无效格分列 |
| B-V9-6 | 预算待更新 | 实际 H 3,242、debug 11,905、总145,382，比用户 1,754/10,235/142,224 多 3,158；probe reserve 另计；无矩阵重复 |

## 9. 证据清单

- `logs/a2_v09/s2_0/reports/matrix-final-v4/report.json`（150 S2 + 6 H；metrics_error=0）
- `logs/a2_v09/s2_0/reports/group_attribution-v1.json` / `.md`
- `logs/a2_v09/s2_0/matrix/progress-matrix-full-r2.jsonl`
- 150 条 `logs/a2_v09/s2_0/matrix/*/r*/{smoke_summary.json,trace.jsonl}`
- 6 条 `logs/a2_v09/s2_0/matrix/h-*/r1/{smoke_summary.json,runner.log,trace.jsonl}`
- `logs/a2_v09/s2_0/cases/h_series_case0.json`（original failure positions）
- `logs/a2_v09/s2_0/evidence/h-q033-r1-{9ac0184ff02d,e826d1832915,ca6621521316}/{smoke_summary.json,trace.jsonl}`
- `logs/a2_v09/s2/probe/1024-r1.log`（构造失败 traceback）
- `logs/a2_v09/s2_0/evidence/q031-platform-inputs-r1/{inputs.npz,inputs_metadata.json,counterfactual_report.json,trace.jsonl}`
- `logs/a2_v09/s2_0/evidence/q031-stairs-inputs-r1/{inputs.npz,inputs_metadata.json,counterfactual_report.json,trace.jsonl}`
- `src/tasks/velocity/v09/{s2_terrain.py,s2_map_ablation.py,s2_resources.py,s2_physics.py}`
- `logs/a2_v09/s2/ledger.jsonl`
- `logs/a2_v09/s2_0/verification/full_matrix_audit_20261122_r9.json`

## 10. 阶段收口（post-rerun）

- S2 全量计数的核心分组与父会话一致；唯一数字不符是 **real w0.8 top 窗 43–55 vs 46–55**。
- 宽 4 m 的“未确认登顶的登阶失败”分类正确；但 0.8/4.0 病例未配对，几何因果不能成立。
- H 三条原 runner_failed 已重跑：H01 得到 physical nonfoot 候选覆盖，H02 成功、H03 更早 nonfoot，后两者不复现；原六例 coverage 仍应按 Q033 记 not_verified，v4 的 covered 状态需改正。
- 边缘诊断的输入合同、valid/age、动作敏感度可复算；初步最大差 0.797 需纠正为 0.7991，并补 invalid 格案例；全幅替换不得用于边缘因果。
- **1024 probe 在 S2TrainEnv 构造时失败**（`_sample_tiles` 绑定参数错误）；无 probe_report/admission，3,709,184 reserve 未释放，训练不得启动，也不能用 512 替代。
- 预算：matrix 130,235、H 原 1,754 与用户一致；H 重跑后 3,242、debug 11,905、总 145,382，比用户 142,224 多 3,158；无矩阵重复/漏记。
